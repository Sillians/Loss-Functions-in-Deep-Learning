""" 
- This loss is commonly used in text-to-image generation and multimodal learning (e.g., CLIP).
- It ensures that matching text–image pairs are close in a joint embedding space, while mismatched pairs are pushed apart.
- Uses cosine similarity as the similarity score.
- Pushes positive pairs closer and negative pairs apart by at least margin.
- Can be extended with contrastive loss (e.g., CLIP’s InfoNCE style).
- Works well for text-to-image generation (AttnGAN, Stable Diffusion with CLIP guidance) and cross-modal retrieval.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 

class TextImageMatchingLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TextImageMatchingLoss, self).__init__()
        self.margin = margin
    
    def forward(self, image_embeds, text_embeds):
        """
        Args:
            image_embeds: Tensor of shape (B, D), image embeddings
            text_embeds: Tensor of shape (B, D), text embeddings
        Returns:
            scalar matching loss
        """
        # Normalize embeddings (cosine similarity requires normalized vectors)
        image_embeds = F.normalize(image_embeds, p=2, dim=1)
        text_embeds = F.normalize(text_embeds, p=2, dim=1)
        
        # Similarity matrix (B x B)
        sim_matrix = torch.matmul(image_embeds, text_embeds.T)
        
        B = sim_matrix.size(0)
        loss = 0.0
        
        for i in range(B):
            # Positive pair similarity
            pos_score = sim_matrix[i, i]
            
            # Negatives: all mismatched pairs
            for j in range(B):
                if i == j:
                    continue
                # Image-to-Text loss
                loss += torch.clamp(self.margin - pos_score + sim_matrix[i, j], min=0)
                # Text-to-Image loss
                loss += torch.clamp(self.margin - pos_score + sim_matrix[j, i], min=0)
        
        # Average over batch
        return loss / (B * (B - 1))
        

# Dummy example: batch of 4, embedding dimension 128
B, D = 4, 128
image_embed = torch.randn(B, D)
text_embed = torch.randn(B, D)

criterion = TextImageMatchingLoss(margin=0.2)
loss = criterion(image_embed, text_embed)
print("Text-Image Matching Loss:", loss.item())