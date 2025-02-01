import os
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from diffusers import AutoencoderKL
import numpy as np

class UnGANableDefense:
    def __init__(self, epsilon=0.1, num_iterations=500, reg_lambda=0.01):
        self.epsilon = epsilon
        self.num_iterations = num_iterations
        self.reg_lambda = reg_lambda
        
        print("Initializing VAE model for feature extraction...")
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", use_safetensors=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.vae = self.vae.to(self.device)
        self.vae.eval()
        
        self.target_size = (512, 512)
        
    def _create_adversarial_patterns(self, size):
        """Create adversarial patterns at the specified size."""
        patterns = []
        H, W = size
        
        # Pattern 1: High-frequency checkerboard
        checker_size = max(H, W) // 64  # Adjust checker size based on image dimensions
        checker = torch.ones((1, 3, H, W), device=self.device)
        for i in range(0, H, checker_size):
            for j in range(0, W, checker_size):
                if (i + j) // checker_size % 2 == 0:
                    checker[0, :, i:min(i+checker_size, H), j:min(j+checker_size, W)] = -1
        patterns.append(checker)
        
        # Pattern 2: Radial gradient
        y, x = torch.meshgrid(torch.linspace(-1, 1, H), 
                            torch.linspace(-1, 1, W))
        radial = torch.sqrt(x**2 + y**2).to(self.device)
        radial = radial.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        patterns.append(radial * 2 - 1)
        
        # Pattern 3: Random high-frequency noise
        noise = torch.randn(1, 3, H // 4, W // 4, device=self.device)
        noise = TF.resize(noise, (H, W))
        patterns.append(noise)
        
        return patterns

    def extract_features(self, x):
        """Extract and normalize VAE features."""
        # Resize to target size for feature extraction
        if x.shape[-2:] != self.target_size:
            x = TF.resize(x, self.target_size, antialias=True)
        
        x = x * 2 - 1
        latent_dist = self.vae.encode(x)
        latent = latent_dist.latent_dist.sample()
        
        B = latent.shape[0]
        latent_flat = latent.view(B, -1)
        latent_flat = latent_flat / (latent_flat.norm(dim=1, keepdim=True) + 1e-10)
        return latent_flat
    
    def compute_pattern_loss(self, perturbed, original_size):
        """Compute loss that encourages alignment with adversarial patterns."""
        # Create patterns at the current image size
        patterns = self._create_adversarial_patterns(original_size)
        
        pattern_losses = []
        feat_perturbed = self.extract_features(perturbed)
        
        for pattern in patterns:
            feat_pattern = self.extract_features(pattern)
            similarity = torch.nn.functional.cosine_similarity(feat_perturbed, feat_pattern, dim=1)
            pattern_losses.append(1 - similarity)
            
        return torch.stack(pattern_losses).mean()

    def compute_loss(self, perturbed, original, delta, iteration, original_size):
        """Compute composite loss combining dissimilarity and pattern alignment."""
        # Original feature dissimilarity
        feat_perturbed = self.extract_features(perturbed)
        feat_original = self.extract_features(original)
        feature_loss = torch.nn.functional.cosine_similarity(feat_perturbed, feat_original, dim=1).mean()
        
        # Pattern alignment loss
        pattern_loss = self.compute_pattern_loss(perturbed, original_size)
        
        # Regularization loss
        reg_loss = torch.mean(delta**2)
        
        # Combine losses with dynamic weighting
        progress = iteration / self.num_iterations
        pattern_weight = 0.3 * (1 - np.cos(progress * np.pi))
        
        total_loss = (0.5 * feature_loss + 
                     pattern_weight * pattern_loss +
                     self.reg_lambda * reg_loss)
        
        return total_loss, feature_loss.item(), pattern_loss.item(), reg_loss.item()

    def protect_image(self, image_path, output_path):
        print(f"\nProcessing image: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        x = TF.to_tensor(image).to(self.device)
        x = x.unsqueeze(0)
        
        # Get input image dimensions
        _, _, H, W = x.shape

        # Initialize with mixture of random noise and pattern-based perturbation
        delta = torch.randn_like(x, device=self.device) * 0.01
        patterns = self._create_adversarial_patterns((H, W))
        for pattern in patterns:
            delta += pattern * 0.01 * torch.randn(1, device=self.device)
        delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        delta.requires_grad_(True)
        
        optimizer = torch.optim.SGD([delta], lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_iterations, eta_min=0.01)
        
        best_loss = float('inf')
        best_delta = None

        for i in range(self.num_iterations):
            optimizer.zero_grad()

            delta_scaled = torch.clamp(delta, -self.epsilon, self.epsilon)
            x_perturbed = torch.clamp(x + delta_scaled, 0, 1)

            total_loss, feat_loss, pattern_loss, reg_loss = self.compute_loss(
                x_perturbed, x, delta_scaled, i, (H, W))

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_delta = delta_scaled.detach().clone()

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                delta.data.clamp_(-self.epsilon, self.epsilon)

            if i % 50 == 0:
                print(f"Iteration {i}: Loss = {total_loss.item():.4f}, "
                      f"Feature Loss = {feat_loss:.4f}, "
                      f"Pattern Loss = {pattern_loss:.4f}, "
                      f"Reg Loss = {reg_loss:.4f}")

        with torch.no_grad():
            final_image = torch.clamp(x + best_delta, 0, 1)
            final_image_pil = TF.to_pil_image(final_image.squeeze(0).cpu())
            final_image_pil = final_image_pil.resize(original_size, Image.Resampling.LANCZOS)
            final_image_pil.save(output_path, 'PNG', quality=100)
            
            perturbation = best_delta.abs()
            print(f"\nOptimization finished. Final loss: {best_loss:.4f}")
            print(f"Protected image saved to: {output_path}")
            print(f"Perturbation stats - Mean: {perturbation.mean():.4f}, "
                  f"Max: {perturbation.max():.4f}")

        return final_image_pil

def test_defense():
    defense = UnGANableDefense(
        epsilon=0.1,
        num_iterations=20,
        reg_lambda=0.01
    )
    test_images = [
        ("backend/image.jpg", "backend/perturbed_result.jpg"),
    ]
    for input_path, output_path in test_images:
        if os.path.exists(input_path):
            defense.protect_image(input_path, output_path)
        else:
            print(f"Input image not found: {input_path}")

if __name__ == "__main__":
    test_defense()