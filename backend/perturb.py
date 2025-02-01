import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from typing import Tuple, Optional, Union
import logging

class FeatureExtractor(nn.Module):
    """Feature extractor using ResNet backbone as specified in the paper"""
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.features.eval()
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.features(x).flatten(1)

class ShadowEncoder(nn.Module):
    """Shadow encoder architecture for GAN inversion"""
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(512)
        
        self.fc = nn.Linear(512 * 8 * 8, latent_dim)
        
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ShadowGenerator(nn.Module):
    """Shadow generator for training shadow encoder"""
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)
        
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        
    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 512, 8, 8)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.relu(self.bn4(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x))
        return x

class UnGANable:
    def __init__(self, 
                 mode: str = 'white-box',
                 latent_dim: int = 512,
                 epsilon: float = 0.05,
                 num_iterations: int = 500,
                 kappa: float = 0.5,
                 device: Optional[str] = None):
        """
        Initialize UnGANable defense system.
        
        Args:
            mode: Defense mode ('white-box', 'gray-box', or 'black-box')
            latent_dim: Dimension of the latent space
            epsilon: Maximum perturbation magnitude
            num_iterations: Number of optimization iterations
            kappa: Trade-off parameter between latent and feature space losses
            device: Computing device (CPU/GPU)
        """
        self.mode = mode
        self.latent_dim = latent_dim
        self.epsilon = epsilon
        self.num_iterations = num_iterations
        self.kappa = kappa
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.feature_extractor = FeatureExtractor().to(self.device)
        
        if mode in ['white-box', 'gray-box']:
            self.shadow_encoder = ShadowEncoder(latent_dim).to(self.device)
            if mode == 'gray-box':
                self.shadow_generator = ShadowGenerator(latent_dim).to(self.device)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.reverse_transform = transforms.Compose([
            transforms.Normalize(mean=[0, 0, 0],
                              std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                              std=[1, 1, 1])
        ])
        
        self.logger = logging.getLogger(__name__)
        
    def train_shadow_encoder(self, 
                           target_encoder: nn.Module,
                           dataloader: DataLoader,
                           num_epochs: int = 100,
                           lr: float = 0.0002):
        """Train shadow encoder to mimic target encoder behavior"""
        if self.mode != 'gray-box':
            raise ValueError("Shadow encoder training only available in gray-box mode")
            
        criterion = nn.MSELoss()
        optimizer_E = torch.optim.Adam(self.shadow_encoder.parameters(), lr=lr)
        optimizer_G = torch.optim.Adam(self.shadow_generator.parameters(), lr=lr)
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_idx, (real_imgs, _) in enumerate(dataloader):
                real_imgs = real_imgs.to(self.device)
                batch_size = real_imgs.size(0)
                
                # Train Generator
                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                optimizer_G.zero_grad()
                
                fake_imgs = self.shadow_generator(z)
                fake_latent = self.shadow_encoder(fake_imgs)
                g_loss = criterion(fake_latent, z)
                
                g_loss.backward()
                optimizer_G.step()
                
                # Train Encoder
                optimizer_E.zero_grad()
                
                with torch.no_grad():
                    target_latent = target_encoder(real_imgs)
                shadow_latent = self.shadow_encoder(real_imgs)
                e_loss = criterion(shadow_latent, target_latent)
                
                e_loss.backward()
                optimizer_E.step()
                
                total_loss += e_loss.item()
                
            avg_loss = total_loss / len(dataloader)
            self.logger.info(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

    def compute_losses(self, 
                      x_perturbed: torch.Tensor,
                      x_original: torch.Tensor,
                      target_latent: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute feature space and latent space losses"""
        # Feature space loss
        feature_loss = -F.cosine_similarity(
            self.feature_extractor(x_perturbed),
            self.feature_extractor(x_original)
        ).mean() + F.mse_loss(
            self.feature_extractor(x_perturbed),
            self.feature_extractor(x_original)
        )
        
        # Latent space loss (if applicable)
        latent_loss = torch.tensor(0.0).to(self.device)
        if self.mode in ['white-box', 'gray-box'] and target_latent is not None:
            perturbed_latent = self.shadow_encoder(x_perturbed)
            if self.mode == 'white-box':
                # Maximize deviation from target latent
                latent_loss = -(F.cosine_similarity(perturbed_latent, target_latent).mean() + 
                              F.mse_loss(perturbed_latent, target_latent))
            else:
                # Push towards zero initialization
                latent_loss = -(F.cosine_similarity(perturbed_latent, 
                                                  torch.zeros_like(perturbed_latent)).mean() +
                              F.mse_loss(perturbed_latent, torch.zeros_like(perturbed_latent)))
        
        return feature_loss, latent_loss

    def protect_image(self, 
                     image_path: str,
                     output_path: str,
                     target_encoder: Optional[nn.Module] = None) -> Image.Image:
        """
        Generate a protected version of the input image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save protected image
            target_encoder: Target encoder model (required for white-box mode)
        """
        if self.mode == 'white-box' and target_encoder is None:
            raise ValueError("Target encoder required for white-box mode")
            
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        x = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get target latent code if in white-box mode
        target_latent = None
        if self.mode == 'white-box':
            with torch.no_grad():
                target_latent = target_encoder(x)
        
        # Initialize perturbation
        delta = torch.zeros_like(x, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=0.01)
        
        # Optimization loop
        for i in range(self.num_iterations):
            optimizer.zero_grad()
            
            x_perturbed = x + delta
            feature_loss, latent_loss = self.compute_losses(x_perturbed, x, target_latent)
            
            # Combined loss
            loss = feature_loss
            if self.mode in ['white-box', 'gray-box']:
                loss = self.kappa * latent_loss + (1 - self.kappa) * feature_loss
            
            loss.backward()
            optimizer.step()
            
            # Project perturbation to epsilon-ball
            with torch.no_grad():
                delta.clamp_(-self.epsilon, self.epsilon)
                x_perturbed = x + delta
                x_perturbed.clamp_(0, 1)
                delta.data = x_perturbed - x
            
            if (i + 1) % 100 == 0:
                self.logger.info(f'Iteration {i+1}/{self.num_iterations}, Loss: {loss.item():.4f}')
        
        # Save protected image
        with torch.no_grad():
            protected_image = x + delta
            protected_image = self.reverse_transform(protected_image.squeeze(0)).cpu()
            protected_image = transforms.ToPILImage()(protected_image)
            protected_image.save(output_path)
            
        return protected_image

def test_defense():
    """Test the UnGANable defense system"""
    # Example usage for black-box mode
    defense_bb = UnGANable(mode='black-box', epsilon=0.05)
    protected_image = defense_bb.protect_image("input.jpg", "protected_bb.jpg")
    
    # Example usage for white-box mode (assuming target_encoder exists)
    defense_wb = UnGANable(mode='white-box', epsilon=0.05)
    target_encoder = ShadowEncoder().eval()  # This would be the actual target encoder
    protected_image = defense_wb.protect_image("input.jpg", "protected_wb.jpg", target_encoder)
    
    print("Protection complete!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_defense()