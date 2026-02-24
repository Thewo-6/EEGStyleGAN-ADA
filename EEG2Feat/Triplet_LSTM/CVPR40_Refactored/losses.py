"""
Loss Functions for Metric Learning

This module implements various loss functions used for learning discriminative
EEG feature representations through metric learning approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for SimCLR-style self-supervised learning.
    
    Implements the NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
    as described in "A Simple Framework for Contrastive Learning of Visual Representations"
    (Chen et al., ICML 2020).
    
    Args:
        batch_size (int): Batch size (pairs count as 2*batch_size after concatenation)
        temperature (float): Temperature parameter for scaling similarities (default: 0.5)
        device (torch.device, optional): Device to place tensors on
    """
    
    def __init__(
        self,
        batch_size: int,
        temperature: float = 0.5,
        device: Optional[torch.device] = None
    ):
        super(ContrastiveLoss, self).__init__()
        
        self.batch_size = batch_size
        self.device = device or torch.device('cpu')
        
        # Register temperature and negative mask as buffers (non-trainable tensors)
        self.register_buffer(
            "temperature",
            torch.tensor(temperature, device=self.device)
        )
        
        # Create mask for negative pairs (all pairs except diagonal)
        self.register_buffer(
            "negatives_mask",
            (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool, device=self.device)).float()
        )
    
    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss between two augmented views.
        
        Args:
            z_i (torch.Tensor): Projection of first augmented view (batch_size, proj_dim)
            z_j (torch.Tensor): Projection of second augmented view (batch_size, proj_dim)
            
        Returns:
            torch.Tensor: Scalar loss value
        """
        # Concatenate positive pairs
        representations = torch.cat([z_i, z_j], dim=0)  # (2*batch_size, proj_dim)
        
        # Compute cosine similarity matrix
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2
        )  # (2*batch_size, 2*batch_size)
        
        # Extract positive pair similarities
        sim_ij = torch.diag(similarity_matrix, self.batch_size)  # (batch_size,)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)  # (batch_size,)
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # (2*batch_size,)
        
        # Compute NT-Xent loss
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
        
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        
        return loss


class TripletMarginLoss(nn.Module):
    """
    Triplet margin loss for metric learning.
    
    Learns embeddings such that positive pairs are close and negative pairs are far apart.
    This is typically used with a triplet mining strategy to select hard triplets.
    
    Args:
        margin (float): Margin for separation between positive and negative pairs (default: 1.0)
        reduction (str): Reduction method - 'mean', 'sum', or 'none' (default: 'mean')
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        reduction: str = 'mean'
    ):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        
        # Use PyTorch's built-in triplet margin loss
        self._loss = nn.TripletMarginLoss(
            margin=margin,
            p=2,  # L2 distance
            swap=False,
            reduction=reduction
        )
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet margin loss.
        
        Args:
            anchor (torch.Tensor): Anchor embeddings (batch_size, embed_dim)
            positive (torch.Tensor): Positive embeddings (batch_size, embed_dim)
            negative (torch.Tensor): Negative embeddings (batch_size, embed_dim)
            
        Returns:
            torch.Tensor: Scalar loss value
        """
        return self._loss(anchor, positive, negative)


class CombinedMetricLoss(nn.Module):
    """
    Combined metric learning loss for flexibility.
    
    Allows weighting and combining multiple metric learning losses.
    
    Args:
        loss_components (dict): Dictionary mapping loss names to (loss_fn, weight) tuples
    """
    
    def __init__(self, loss_components: dict):
        super(CombinedMetricLoss, self).__init__()
        self.losses = nn.ModuleDict()
        self.weights = {}
        
        for name, (loss_fn, weight) in loss_components.items():
            self.losses[name] = loss_fn
            self.weights[name] = weight
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Compute combined loss.
        
        Returns:
            torch.Tensor: Weighted sum of all component losses
        """
        total_loss = torch.tensor(0.0, device=args[0].device)
        
        for name, loss_fn in self.losses.items():
            component_loss = loss_fn(*args, **kwargs)
            weighted_loss = self.weights[name] * component_loss
            total_loss = total_loss + weighted_loss
        
        return total_loss


if __name__ == '__main__':
    print("Testing loss functions...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    proj_dim = 128
    
    # Test ContrastiveLoss
    print("\n1. Testing ContrastiveLoss:")
    contrastive_loss = ContrastiveLoss(batch_size=batch_size, device=device)
    z_i = F.normalize(torch.randn(batch_size, proj_dim, device=device), dim=1)
    z_j = F.normalize(torch.randn(batch_size, proj_dim, device=device), dim=1)
    loss = contrastive_loss(z_i, z_j)
    print(f"   Loss value: {loss.item():.4f}")
    
    # Test TripletMarginLoss
    print("\n2. Testing TripletMarginLoss:")
    triplet_loss = TripletMarginLoss(margin=1.0)
    anchor = F.normalize(torch.randn(batch_size, proj_dim), dim=1)
    positive = anchor + 0.1 * torch.randn(batch_size, proj_dim)  # Similar to anchor
    negative = torch.randn(batch_size, proj_dim)  # Random negative
    positive = F.normalize(positive, dim=1)
    negative = F.normalize(negative, dim=1)
    loss = triplet_loss(anchor, positive, negative)
    print(f"   Loss value: {loss.item():.4f}")
