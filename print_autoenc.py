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
train_dataset = CustomImageDataset(root=data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Assuming you have a test set as well in another folder
test_dataset = CustomImageDataset(root=data_dir, transform=transform)
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


def load_model(checkpoint_path, model):
    """Load the trained model from checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def visualize_reconstructions(model, dataloader, num_images=8, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Visualize original and reconstructed images side by side."""
    # Get a batch of images
    images, _ = next(iter(dataloader))
    images = images[:num_images].to(device)
    
    # Generate reconstructions
    with torch.no_grad():
        reconstructed = model(images)
    
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    
    images = images * std + mean
    reconstructed = reconstructed * std + mean
    
    # Create figure
    fig, axes = plt.subplots(2, num_images, figsize=(20, 5))
    
    # Move tensors to CPU and convert to numpy
    images = images.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    
    for i in range(num_images):
        # Original images
        orig_img = np.transpose(images[i], (1, 2, 0))
        orig_img = np.clip(orig_img, 0, 1)  # Clip values to valid range
        axes[0, i].imshow(orig_img)
        axes[0, i].axis('off')
        axes[0, i].set_title('Original')
        
        # Reconstructed images
        recon_img = np.transpose(reconstructed[i], (1, 2, 0))
        recon_img = np.clip(recon_img, 0, 1)  # Clip values to valid range
        axes[1, i].imshow(recon_img)
        axes[1, i].axis('off')
        axes[1, i].set_title('Reconstructed')
    
    plt.tight_layout()
    plt.show()

def reconstruct_single_image(model, image_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Reconstruct a single image from path."""
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Generate reconstruction
    model.eval()
    with torch.no_grad():
        reconstructed = model(image_tensor)
    
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    
    image_tensor = image_tensor * std + mean
    reconstructed = reconstructed * std + mean
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Display original
    orig_img = np.transpose(image_tensor[0].cpu().numpy(), (1, 2, 0))
    orig_img = np.clip(orig_img, 0, 1)
    ax1.imshow(orig_img)
    ax1.axis('off')
    ax1.set_title('Original')
    
    # Display reconstruction
    recon_img = np.transpose(reconstructed[0].cpu().numpy(), (1, 2, 0))
    recon_img = np.clip(recon_img, 0, 1)
    ax2.imshow(recon_img)
    ax2.axis('off')
    ax2.set_title('Reconstructed')
    
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    # Initialize the model (using the same architecture as during training)
    model = ViTAutoencoder(
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        encoder_depth=6,
        decoder_depth=6,
        num_heads=8
    )
    
    # Load the trained weights
    checkpoint_path = "checkpoints/best_model.pth"
    model = load_model(checkpoint_path, model)
    """ 
    #Visualize reconstructions from test dataset
    print("Visualizing reconstructions from test dataset...")
    visualize_reconstructions(model, test_loader)
    """
    
    # Example 2: Reconstruct a single image
    print("\nReconstructing single image...")
    image_path = "./random_imagenet/test/ILSVRC2012_val_00045008.JPEG"  # Replace with your image path
    if os.path.exists(image_path):
        reconstruct_single_image(model, image_path)
    else:
        print(f"Image not found at {image_path}")
    



"""

    # Calculate reconstruction error on test set
    print("\nCalculating average reconstruction error on test set...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.MSELoss()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            reconstructed = model(images)
            loss = criterion(reconstructed, images)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    print(f"Average reconstruction MSE on test set: {avg_loss:.6f}")


    """