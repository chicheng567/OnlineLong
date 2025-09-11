"""
HierarchicalMemoryBank implementation for VideoLLaMA3.
Adapted from VideoChatOnline's memory bank system with feature adaptation for SigLIP encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class MemoryBankConfig:
    """Configuration for HierarchicalMemoryBank."""
    capacities: List[int] = None  # Capacity for each memory group
    reduced_sizes: List[int] = None  # Token reduction size for each group
    feature_dim: int = 1152  # SigLIP feature dimension (vs 3200 for InternVision)
    enable_compression: bool = True
    similarity_threshold: float = 0.8
    
    def __post_init__(self):
        if self.capacities is None:
            self.capacities = [8, 4, 1]  # Default: short, mid, long-term
        if self.reduced_sizes is None:
            self.reduced_sizes = [144, 64, 36]  # Progressive reduction


class FeatureAdapter(nn.Module):
    """Adapter layer to handle feature dimension differences between vision encoders."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (input_dim + output_dim) // 2
            
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.adapter(x)


class HierarchicalMemoryBank:
    """
    Hierarchical Memory Bank for online video understanding.
    
    Manages multiple memory groups with different capacities and compression ratios:
    - Short-term memory: High capacity, minimal compression
    - Mid-term memory: Medium capacity, moderate compression  
    - Long-term memory: Low capacity, high compression
    
    Features:
    - Similarity-based frame eviction when capacity is reached
    - Progressive token reduction for efficient memory usage
    - Feature adaptation for different vision encoder dimensions
    """
    
    def __init__(self, config: MemoryBankConfig):
        self.config = config
        self.groups = [
            {
                "tokens": [], 
                "capacity": cap, 
                "reduced_size": size,
                "name": ["short_term", "mid_term", "long_term"][i]
            }
            for i, (cap, size) in enumerate(zip(config.capacities, config.reduced_sizes))
        ]
        
        # Feature adapter for SigLIP -> processing dimensions
        self.feature_adapter = FeatureAdapter(
            input_dim=config.feature_dim,
            output_dim=config.feature_dim
        ) if config.feature_dim != 1152 else None
        
    def _adapt_features(self, tokens: torch.Tensor) -> torch.Tensor:
        """Adapt features if needed for dimension compatibility."""
        if self.feature_adapter is not None:
            original_shape = tokens.shape
            tokens = tokens.view(-1, original_shape[-1])
            tokens = self.feature_adapter(tokens)
            tokens = tokens.view(original_shape)
        return tokens
        
    def _meanpool(self, tokens: torch.Tensor) -> torch.Tensor:
        """Compute mean pooling over spatial dimensions."""
        if tokens.dim() == 4:  # [B, C, H, W]
            return tokens.mean(dim=(2, 3))
        elif tokens.dim() == 3:  # [B, L, C] - sequence format
            return tokens.mean(dim=1)
        else:
            return tokens
            
    def _find_most_similar_frame(self, group: List[Dict]) -> int:
        """Find the pair of most similar frames in a group for eviction."""
        if len(group) < 2:
            return 0
            
        cls_tokens = torch.stack([self._meanpool(item["tokens"]) for item in group])
        
        # Compute pairwise similarities
        n = cls_tokens.size(0)
        max_similarity = -1
        most_similar_idx = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = F.cosine_similarity(
                    cls_tokens[i:i+1], 
                    cls_tokens[j:j+1], 
                    dim=-1
                ).item()
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    # Return the more recent frame (higher index)
                    most_similar_idx = max(i, j)
        
        return most_similar_idx
        
    def _reduce_tokens(self, tokens: torch.Tensor, target_size: int) -> torch.Tensor:
        """Reduce token resolution through interpolation."""
        if not self.config.enable_compression:
            return tokens
            
        if tokens.dim() == 3:  # [B, L, C] format
            # For sequence format, use adaptive pooling
            if tokens.size(1) <= target_size:
                return tokens
            pooled = F.adaptive_avg_pool1d(
                tokens.transpose(1, 2), 
                target_size
            ).transpose(1, 2)
            return pooled
            
        elif tokens.dim() == 4:  # [B, C, H, W] format
            H = W = int(target_size ** 0.5)
            if tokens.size(-2) * tokens.size(-1) <= target_size:
                return tokens
            return F.interpolate(
                tokens, 
                size=(H, W), 
                mode="bilinear", 
                align_corners=False
            )
        else:
            return tokens
            
    def _update_group(
        self, 
        group: Dict, 
        new_tokens: torch.Tensor, 
        index: int, 
        cls_token: torch.Tensor, 
        next_group: Optional[Dict] = None
    ):
        """Update a memory group with new tokens, handling capacity overflow."""
        # Handle capacity overflow
        if len(group["tokens"]) >= group["capacity"] and group["capacity"] > 0:
            # Find most similar frame for eviction
            similar_frame_idx = self._find_most_similar_frame(group["tokens"])
            evicted_item = group["tokens"][similar_frame_idx]
            
            # Move evicted item to next group if available
            if next_group is not None:
                reduced_tokens = self._reduce_tokens(
                    evicted_item["tokens"],
                    target_size=next_group["reduced_size"]
                )
                self._update_group(
                    next_group,
                    reduced_tokens,
                    evicted_item["index"],
                    evicted_item["cls_token"]
                )
            
            # Remove evicted item from current group
            group["tokens"].pop(similar_frame_idx)
            
        # Add new item to group
        reduced_tokens = self._reduce_tokens(new_tokens, target_size=group["reduced_size"])
        group["tokens"].append({
            "tokens": reduced_tokens,
            "index": index,
            "cls_token": cls_token,
            "timestamp": getattr(self, '_current_timestamp', None)
        })
        
    def update_memory(
        self, 
        new_tokens: torch.Tensor, 
        index: int, 
        cls_token: Optional[torch.Tensor] = None,
        timestamp: Optional[float] = None
    ):
        """
        Update memory bank with new frame tokens.
        
        Args:
            new_tokens: Feature tokens from new frame [B, C, H, W] or [B, L, C]
            index: Frame index for tracking
            cls_token: Optional CLS token for similarity computation
            timestamp: Optional timestamp for temporal tracking
        """
        self._current_timestamp = timestamp
        
        # Adapt features if needed
        new_tokens = self._adapt_features(new_tokens)
        
        # Generate CLS token if not provided
        if cls_token is None:
            cls_token = self._meanpool(new_tokens)
            
        # Find appropriate group based on token size (start with short-term)
        target_group_idx = 0
        for i, group in enumerate(self.groups):
            if new_tokens.numel() >= group["reduced_size"]:
                target_group_idx = i
                break
                
        # Update the target group
        group = self.groups[target_group_idx]
        next_group = (
            self.groups[target_group_idx + 1] 
            if target_group_idx + 1 < len(self.groups) 
            else None
        )
        
        self._update_group(group, new_tokens, index, cls_token, next_group)
        
    def get_memory_summary(self) -> Dict:
        """Get summary of current memory bank state."""
        summary = {}
        for group in self.groups:
            summary[group["name"]] = {
                "count": len(group["tokens"]),
                "capacity": group["capacity"],
                "utilization": len(group["tokens"]) / max(group["capacity"], 1),
                "indices": [item["index"] for item in group["tokens"]],
            }
        return summary
        
    def retrieve_memory(
        self, 
        query_features: torch.Tensor, 
        top_k: int = 5,
        similarity_threshold: float = None
    ) -> List[Dict]:
        """
        Retrieve most relevant memory frames based on query features.
        
        Args:
            query_features: Query feature tensor
            top_k: Number of top similar frames to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of retrieved memory items with similarity scores
        """
        if similarity_threshold is None:
            similarity_threshold = self.config.similarity_threshold
            
        query_cls = self._meanpool(query_features)
        all_items = []
        
        # Collect all memory items with similarities
        for group in self.groups:
            for item in group["tokens"]:
                item_cls = item["cls_token"]
                similarity = F.cosine_similarity(
                    query_cls.unsqueeze(0), 
                    item_cls.unsqueeze(0), 
                    dim=-1
                ).item()
                
                if similarity >= similarity_threshold:
                    all_items.append({
                        "item": item,
                        "similarity": similarity,
                        "group_name": group["name"]
                    })
                    
        # Sort by similarity and return top-k
        all_items.sort(key=lambda x: x["similarity"], reverse=True)
        return all_items[:top_k]
        
    def clear_memory(self):
        """Clear all memory groups."""
        for group in self.groups:
            group["tokens"].clear()
            
    def get_total_memory_size(self) -> int:
        """Get total number of stored frames across all groups."""
        return sum(len(group["tokens"]) for group in self.groups)
        
    def export_memory_state(self) -> Dict:
        """Export memory state for checkpointing."""
        return {
            "config": self.config,
            "groups": [
                {
                    "name": group["name"],
                    "capacity": group["capacity"],
                    "reduced_size": group["reduced_size"],
                    "tokens": [
                        {
                            "tokens": item["tokens"].cpu(),
                            "index": item["index"],
                            "cls_token": item["cls_token"].cpu(),
                            "timestamp": item.get("timestamp")
                        }
                        for item in group["tokens"]
                    ]
                }
                for group in self.groups
            ]
        }
        
    @classmethod
    def from_exported_state(cls, state: Dict, device: str = "cuda"):
        """Restore memory bank from exported state."""
        memory_bank = cls(state["config"])
        
        for group_state, group in zip(state["groups"], memory_bank.groups):
            group["tokens"] = [
                {
                    "tokens": item["tokens"].to(device),
                    "index": item["index"],
                    "cls_token": item["cls_token"].to(device),
                    "timestamp": item.get("timestamp")
                }
                for item in group_state["tokens"]
            ]
            
        return memory_bank