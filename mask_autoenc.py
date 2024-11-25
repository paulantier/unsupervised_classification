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

# Define transformations (ImageNet pre-processing)
transform = transforms.Compose([
    transforms.Resize(256),           # Resize the shorter side to 256
    transforms.CenterCrop(224),       # Crop the image to 224x224
    transforms.ToTensor(),            # Convert the image to a tensor
    transforms.Normalize(             # Normalize using ImageNet mean and std
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
])

# Path to the folder containing JPEG images
data_dir = './random_imagenet'

# Create dataset and dataloader
train_dataset = CustomImageDataset(root=data_dir+"/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Assuming you have a test set as well in another folder
test_dataset = CustomImageDataset(root=data_dir+"/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)



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

# Example usage
if __name__ == "__main__":
    # Create model
    model = ViTAutoencoder(
        img_size=224,
        patch_size=32,
        in_channels=3,
        embed_dim=768,
        encoder_depth=6,
        decoder_depth=6,
        num_heads=8
    )
    
def train_vit_autoencoder(
    model,
    train_loader,
    test_loader,
    num_epochs=100,
    learning_rate=1e-4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_dir="checkpoints",
):


    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Move model to device
    model = model.to(device)
    
    # Initialize optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Keep track of best loss
    best_test_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch_idx, (images, _) in enumerate(train_pbar):
            images = images.to(device)
            
            # Forward pass
            reconstructed = model(images)
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
                reconstructed = model(images)
                loss = criterion(reconstructed, images)
                test_loss += loss.item()
                test_pbar.set_postfix({'loss': loss.item()})
                
                # Save sample reconstructions periodically
                if batch_idx == 0 and epoch % 5 == 0:
                    save_sample_reconstructions(
                        images, 
                        reconstructed, 
                        epoch, 
                        save_dir,
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

def save_sample_reconstructions(original, reconstructed, epoch, save_dir):
    """Save and log sample reconstructions."""
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(original.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(original.device)
    
    original = original * std + mean
    reconstructed = reconstructed * std + mean
    
    # Convert to numpy and move to CPU
    original = original.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i in range(4):
        # Original images
        axes[0, i].imshow(np.transpose(original[i], (1, 2, 0)))
        axes[0, i].axis('off')
        axes[0, i].set_title('Original')
        
        # Reconstructed images
        axes[1, i].imshow(np.transpose(reconstructed[i], (1, 2, 0)))
        axes[1, i].axis('off')
        axes[1, i].set_title('Reconstructed')
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, f'reconstruction_epoch_{epoch}.png')
    plt.savefig(save_path)
    plt.close()
    

def load_model(checkpoint_path, model):
    """Load the trained model from checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

# Example usage
if __name__ == "__main__":
    import numpy as np
    
    # Initialize model (using the ViTAutoencoder from previous code)
    model = ViTAutoencoder(
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        encoder_depth=6,
        decoder_depth=6,
        num_heads=8
    )

    checkpoint_path = "checkpoints/best_model.pth"
    model = load_model(checkpoint_path, model)
    
    # Train the model
    train_vit_autoencoder(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=100,
        learning_rate=1e-4,
    )