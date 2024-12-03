import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition
from tqdm import tqdm
from torchvision.transforms import functional as F
import os
from PIL import Image


class CustomImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = [os.path.join(root, fname) for fname in os.listdir(root) if fname.endswith('.JPEG') or fname.endswith('jpg')]
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Ensure RGB mode
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0

transform = transforms.Compose([
    transforms.Resize((160, 160)),       # Slightly upscale to give cropping room
    transforms.RandomCrop(128),         # Randomly crop to 64x64 for augmentation
    transforms.RandomHorizontalFlip(), # Randomly flip the image horizontally
    transforms.ColorJitter(            # Adjust brightness, contrast, saturation
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
    ),
    transforms.RandomGrayscale(p=0.1),# Occasionally convert to grayscale
    transforms.ToTensor(),            # Convert to tensor
    transforms.Normalize(             # Normalize using ImageNet mean and std
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
])

# Path to the folder containing JPEG images
data_dir = './random_imagenet'

# Create dataset and dataloader
train_dataset = CustomImageDataset(root=data_dir+"/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

# Assuming you have a test set as well in another folder
test_dataset = CustomImageDataset(root=data_dir+"/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)



class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # (B, E, H', W')
        x = x.flatten(2)  # (B, E, N)
        x = x.transpose(1, 2)  # (B, N, E)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=6, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                batch_first=True
            )
            for _ in range(depth)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=768, depth=6, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                batch_first=True
            )
            for _ in range(depth)
        ])
        
    def forward(self, x, memory):
        for layer in self.layers:
            x = layer(x, memory)
        return x

class ViTAutoencoder(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        encoder_depth=6,
        decoder_depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()
        
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Add position embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, (img_size // patch_size) ** 2, embed_dim)
        )
        
        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        self.decoder = TransformerDecoder(
            embed_dim=embed_dim,
            depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        # Reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(embed_dim, patch_size * patch_size * in_channels),
            nn.GELU()
        )
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels

    def unpatchify(self, x):
        """Convert patched representation back to image."""
        B, N, P = x.shape  # batch, num_patches, patch_dim
        h = w = int(N ** 0.5)
        p = self.patch_size
        c = self.in_channels
        
        x = x.reshape(B, h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(B, c, h * p, w * p)
        return imgs

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Encode
        encoded = self.encoder(x)
        
        # Generate decoder query embeddings (could be learned or copied from encoder)
        query = self.pos_embed.expand(x.shape[0], -1, -1)
        
        # Decode
        decoded = self.decoder(query, encoded)
        
        # Reconstruct patches
        patches = self.reconstruction_head(decoded)
        
        # Reshape patches to image
        output = self.unpatchify(patches)
        
        return output


def quadtree_compression(image, threshold):
    """
    Quadtree compression and decompression for batched images with channels.
    Args:
        image: Tensor of shape (batch_size, channels, height, width).
        threshold: Variance threshold for determining uniform regions.
    Returns:
        decompressed: Tensor of the same shape as the input image.
    """
    def is_uniform(region):
        return torch.var(region, dim=(-2, -1), keepdim=True) < threshold

    def divide_and_conquer(region, size):
        if size == 1 or is_uniform(region).all():
            return torch.mean(region, dim=(-2, -1), keepdim=True).expand(-1, -1, size, size)

        half = size // 2
        # Subdivide the region into quadrants
        quads = [
            region[:, :, :half, :half],  # Top-left
            region[:, :, :half, half:],  # Top-right
            region[:, :, half:, :half],  # Bottom-left
            region[:, :, half:, half:],  # Bottom-right
        ]

        results = [divide_and_conquer(quad, half) for quad in quads]

        # Combine quadrants back into one region
        top = torch.cat(results[:2], dim=-1)
        bottom = torch.cat(results[2:], dim=-1)
        return torch.cat([top, bottom], dim=-2)

    # Initial call to the recursive divide_and_conquer function
    batch_size, channels, height, width = image.shape

    decompressed = divide_and_conquer(image, height)

    return decompressed


def noise_image(images, std_dev=0.05):
    noise = torch.randn_like(images) * std_dev
    return images + noise
    
def train_vit_autoencoder(
    model,
    train_loader,
    test_loader,
    num_epochs=100,
    learning_rate=1e-4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_dir="qt_vit_checkpoints",
    start_epoch=1,
):


    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Move model to device
    model = model.to(device)
    
    # Initialize optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Keep track of best loss
    best_test_loss = float('inf')
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch_idx, (images, _) in enumerate(train_pbar):
            images = images.to(device)

            qt_images = quadtree_compression(images.clone(), threshold=1)
            noisy_qt_images = noise_image(qt_images, std_dev=0.2)
            
            # Forward pass
            reconstructed = model(noisy_qt_images)
            loss = criterion(reconstructed, images)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})

        
        # Calculate average training loss for the epoch
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        test_loss = 0
        test_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Test]')
        
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(test_pbar):
                images = images.to(device)
                qt_images = quadtree_compression(images.clone(), threshold=1)
                noisy_qt_images = noise_image(qt_images, std_dev=0.2)
                
                reconstructed = model(noisy_qt_images)
                loss = criterion(reconstructed, images)
                test_loss += loss.item()
                test_pbar.set_postfix({'loss': loss.item()})
                
                # Save sample reconstructions periodically
                if (batch_idx <=2 and (epoch+1) % 2 == 0):
                    save_sample_reconstructions(
                        images, 
                        qt_images, 
                        noisy_qt_images, 
                        reconstructed, 
                        epoch, 
                        save_dir,
                        batch_idx,
                    )
        
        # Calculate average test loss
        avg_test_loss = test_loss / len(test_loader)
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch results
        print(f'Average Train Loss: {avg_train_loss:.6f}')
        print(f'Average Test Loss: {avg_test_loss:.6f}')
        
        
        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'test_loss': avg_test_loss,
            }, os.path.join(save_dir, 'best_model.pth'))

def save_sample_reconstructions(original, noisy, masked_noisy, reconstructed, epoch, save_dir, batch_number):
    """Save and log sample reconstructions."""
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(original.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(original.device)
    
    original = original * std + mean
    reconstructed = reconstructed * std + mean
    noisy = noisy * std + mean
    masked_noisy = masked_noisy * std + mean
    
    # Convert to numpy and move to CPU
    original = original.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    noisy = noisy.cpu().numpy()
    masked_noisy = masked_noisy.cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    
    for i in range(4):
        # Original images
        axes[0, i].imshow(np.transpose(original[i], (1, 2, 0)))
        axes[0, i].axis('off')
        axes[0, i].set_title('Original')
        
        # Reconstructed images
        axes[1, i].imshow(np.transpose(noisy[i], (1, 2, 0)))
        axes[1, i].axis('off')
        axes[1, i].set_title('QuadTree Compression')

        # Reconstructed images
        axes[2, i].imshow(np.transpose(masked_noisy[i], (1, 2, 0)))
        axes[2, i].axis('off')
        axes[2, i].set_title('QuadTree + noise')

        # Reconstructed images
        axes[3, i].imshow(np.transpose(reconstructed[i], (1, 2, 0)))
        axes[3, i].axis('off')
        axes[3, i].set_title('Reconstructed')
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, f'Reconstruction_epoch{epoch+1}_batch{batch_number}.png')
    plt.savefig(save_path)
    plt.close()
    

def load_model(checkpoint_path, model):
    """Load the trained model and epoch from a checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Retrieve the epoch information
    epoch = checkpoint.get('epoch', 0) + 1  # Use .get() to handle cases where 'epoch' might not be present
    
    return model, epoch

# Example usage
if __name__ == "__main__":
    import numpy as np
    
    # Initialize model (using the ViTAutoencoder from previous code)
    
    model = ViTAutoencoder(
        img_size=128,          
        patch_size=16,         
        in_channels=3,         
        embed_dim=1024,        
        num_heads=16,          
        encoder_depth=12,      
        decoder_depth=8        # Slightly fewer layers in the decoder (fewer are often needed for reconstruction).
    )
    """
    model = ViTAutoencoder(
        img_size=64,
        patch_size=8,
        in_channels=3,
        embed_dim=768,
        encoder_depth=6,
        decoder_depth=4,
        num_heads=8
    ) 
    """

    checkpoint_path = "qt_vit_checkpoints/best_model.pth"
    start_epoch=0
    #model, start_epoch = load_model(checkpoint_path, model)

    
    # Train the model
    train_vit_autoencoder(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=100,
        learning_rate=1e-4,
        start_epoch = start_epoch,
    )