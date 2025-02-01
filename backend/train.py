import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import yaml

from unganable import UnGANable, ShadowEncoder

class WhiteBoxTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.target_encoder = self._create_target_encoder()
        self.defense = UnGANable(
            mode='white-box',
            latent_dim=config['latent_dim'],
            epsilon=config['epsilon'],
            device=self.device
        )
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.defense.shadow_encoder.parameters(),
            lr=config['learning_rate'],
            betas=(config['beta1'], config['beta2'])
        )
        
        # Setup data
        self.train_loader, self.val_loader = self._setup_data()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
    def _create_target_encoder(self):
        """Create and load target encoder"""
        target_encoder = ShadowEncoder(self.config['latent_dim']).to(self.device)
        if self.config['target_encoder_path']:
            state_dict = torch.load(self.config['target_encoder_path'])
            target_encoder.load_state_dict(state_dict)
        target_encoder.eval()
        return target_encoder
    
    def _setup_data(self):
        """Setup training and validation data loaders"""
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Load dataset
        full_dataset = datasets.ImageFolder(self.config['data_dir'], transform=transform)
        
        # Split into train/val
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers']
        )
        
        return train_loader, val_loader
    
    def save_checkpoint(self, epoch, loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'shadow_encoder_state_dict': self.defense.shadow_encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }
        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        self.logger.info(f'Saved checkpoint: {path}')
        
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        self.defense.shadow_encoder.load_state_dict(checkpoint['shadow_encoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']
    
    def train_epoch(self):
        """Train for one epoch"""
        self.defense.shadow_encoder.train()
        total_loss = 0
        
        for batch_idx, (images, _) in enumerate(tqdm(self.train_loader)):
            images = images.to(self.device)
            
            # Get target latent codes
            with torch.no_grad():
                target_latent = self.target_encoder(images)
            
            # Train shadow encoder
            self.optimizer.zero_grad()
            shadow_latent = self.defense.shadow_encoder(images)
            
            # Compute losses (cosine similarity and MSE as per paper)
            cos_loss = -torch.nn.functional.cosine_similarity(
                shadow_latent, target_latent, dim=1
            ).mean()
            mse_loss = torch.nn.functional.mse_loss(shadow_latent, target_latent)
            loss = cos_loss + mse_loss
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % self.config['log_interval'] == 0:
                self.logger.info(f'Batch [{batch_idx + 1}/{len(self.train_loader)}], Loss: {loss.item():.4f}')
                
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate model"""
        self.defense.shadow_encoder.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, _ in self.val_loader:
                images = images.to(self.device)
                target_latent = self.target_encoder(images)
                shadow_latent = self.defense.shadow_encoder(images)
                
                cos_loss = -torch.nn.functional.cosine_similarity(
                    shadow_latent, target_latent, dim=1
                ).mean()
                mse_loss = torch.nn.functional.mse_loss(shadow_latent, target_latent)
                loss = cos_loss + mse_loss
                
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        # Load checkpoint if specified
        start_epoch = 0
        if self.config['resume_checkpoint']:
            start_epoch = self.load_checkpoint(self.config['resume_checkpoint'])
            self.logger.info(f"Resumed from checkpoint at epoch {start_epoch}")
        
        best_val_loss = float('inf')
        
        for epoch in range(start_epoch, self.config['num_epochs']):
            self.logger.info(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            
            # Train
            train_loss = self.train_epoch()
            self.logger.info(f'Training Loss: {train_loss:.4f}')
            
            # Validate
            val_loss = self.validate()
            self.logger.info(f'Validation Loss: {val_loss:.4f}')
            
            # Save checkpoint
            if (epoch + 1) % self.config['checkpoint_interval'] == 0:
                self.save_checkpoint(epoch + 1, val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    self.defense.shadow_encoder.state_dict(),
                    self.checkpoint_dir / 'best_model.pt'
                )
                self.logger.info('Saved best model')

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Train UnGANable white-box defense')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize and run trainer
    trainer = WhiteBoxTrainer(config)
    trainer.train()

if __name__ == '__main__':
    main()