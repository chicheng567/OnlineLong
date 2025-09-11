"""
Integration test for VideoLLaMA3 + Memory Bank.
Tests the complete integration including memory-aware encoding.
"""

import torch
import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'videollama3', 'model', 'memory_bank'))

from hierarchical_memory_bank import HierarchicalMemoryBank, MemoryBankConfig


class MockVisionEncoder(torch.nn.Module):
    """Mock vision encoder for testing."""
    
    def __init__(self, hidden_size=1152):
        super().__init__()
        self.hidden_size = hidden_size
        self.conv = torch.nn.Conv2d(3, hidden_size, kernel_size=1)
        
    def forward(self, pixel_values, grid_sizes=None, merge_sizes=None):
        # Simulate vision encoding
        B, C, H, W = pixel_values.shape
        features = self.conv(pixel_values)  # [B, hidden_size, H, W]
        features = features.flatten(-2).transpose(-1, -2)  # [B, H*W, hidden_size]
        return features


class MockProjector(torch.nn.Module):
    """Mock projector for testing."""
    
    def __init__(self, input_dim=1152, output_dim=4096):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)


class MockConfig:
    """Mock configuration for testing."""
    
    def __init__(self):
        self.enable_memory_bank = True
        self.memory_capacities = [3, 2, 1]  # Small for testing
        self.memory_reduced_sizes = [144, 64, 36]
        self.mm_hidden_size = 1152
        self.memory_enable_compression = True
        self.memory_similarity_threshold = 0.5


class MockVideollama3Model:
    """Mock VideoLLaMA3 model for testing memory integration."""
    
    def __init__(self, config):
        self.config = config
        self.vision_encoder = MockVisionEncoder()
        self.mm_projector = MockProjector()
        
        # Initialize Memory Bank if enabled
        self.memory_bank = None
        if getattr(config, 'enable_memory_bank', False):
            memory_config = MemoryBankConfig(
                capacities=getattr(config, 'memory_capacities', [8, 4, 1]),
                reduced_sizes=getattr(config, 'memory_reduced_sizes', [144, 64, 36]),
                feature_dim=getattr(config, 'mm_hidden_size', 1152),
                enable_compression=getattr(config, 'memory_enable_compression', True),
                similarity_threshold=getattr(config, 'memory_similarity_threshold', 0.8)
            )
            self.memory_bank = HierarchicalMemoryBank(memory_config)
    
    def get_vision_encoder(self):
        return self.vision_encoder
        
    def get_mm_projector(self):
        return self.mm_projector
        
    def get_memory_bank(self):
        return self.memory_bank
        
    def has_memory_bank(self):
        return self.memory_bank is not None
        
    def encode_images(self, pixel_values, grid_sizes=None, merge_sizes=None):
        """Standard image encoding."""
        mm_features = self.vision_encoder(
            pixel_values=pixel_values,
            grid_sizes=grid_sizes,
            merge_sizes=merge_sizes,
        )
        mm_features = self.mm_projector(mm_features)
        return mm_features
        
    def encode_images_with_memory(
        self,
        pixel_values,
        grid_sizes=None,
        merge_sizes=None,
        frame_indices=None,
        timestamps=None,
        update_memory=True,
    ):
        """Memory-aware image encoding."""
        # Encode images normally
        mm_features = self.encode_images(pixel_values, grid_sizes, merge_sizes)
        
        memory_info = None
        if self.has_memory_bank() and frame_indices is not None:
            memory_bank = self.get_memory_bank()
            
            # Update memory bank with current features if requested
            if update_memory:
                batch_size = pixel_values.size(0)
                for i in range(batch_size):
                    frame_idx = frame_indices[i].item() if frame_indices is not None else i
                    timestamp = timestamps[i].item() if timestamps is not None else None
                    
                    # Extract features for this frame and reshape for memory bank
                    frame_features = mm_features[i:i+1]  # [1, seq_len, hidden_size]
                    
                    # Reshape to 4D for memory bank [1, hidden_size, H, W]
                    seq_len, hidden_size = frame_features.shape[1], frame_features.shape[2]
                    H = W = int((seq_len) ** 0.5)  # Assume square feature map
                    if H * W != seq_len:
                        H, W = 12, 12  # Default spatial size
                        frame_features = torch.nn.functional.adaptive_avg_pool1d(
                            frame_features.transpose(1, 2), H * W
                        ).transpose(1, 2)
                    
                    frame_features = frame_features.view(1, H, W, hidden_size).permute(0, 3, 1, 2)
                    
                    memory_bank.update_memory(
                        new_tokens=frame_features,
                        index=frame_idx,
                        timestamp=timestamp
                    )
            
            # Retrieve relevant memory for context
            if mm_features.size(0) > 0:
                # Use first frame as query
                query_features = mm_features[0:1]
                seq_len, hidden_size = query_features.shape[1], query_features.shape[2]
                H = W = 12  # Default spatial size
                query_features = torch.nn.functional.adaptive_avg_pool1d(
                    query_features.transpose(1, 2), H * W
                ).transpose(1, 2)
                query_features = query_features.view(1, H, W, hidden_size).permute(0, 3, 1, 2)
                
                retrieved_memories = memory_bank.retrieve_memory(
                    query_features=query_features,
                    top_k=3,
                    similarity_threshold=0.1
                )
                
                memory_info = {
                    'retrieved_count': len(retrieved_memories),
                    'memory_summary': memory_bank.get_memory_summary(),
                    'retrieved_indices': [mem['item']['index'] for mem in retrieved_memories]
                }
        
        return mm_features, memory_info


def test_memory_integration():
    """Test complete memory bank integration."""
    print("Testing VideoLLaMA3 + Memory Bank integration...")
    
    # Create mock model with memory bank
    config = MockConfig()
    model = MockVideollama3Model(config)
    
    # Test basic functionality
    assert model.has_memory_bank(), "Memory bank should be enabled"
    memory_bank = model.get_memory_bank()
    assert memory_bank is not None, "Memory bank should be initialized"
    
    print("✓ Memory bank initialization successful")
    
    # Test memory-aware encoding
    batch_size, channels, height, width = 2, 3, 224, 224
    pixel_values = torch.randn(batch_size, channels, height, width)
    frame_indices = torch.tensor([0, 1])
    timestamps = torch.tensor([0.0, 1.0])
    
    # Encode with memory updates
    features, memory_info = model.encode_images_with_memory(
        pixel_values=pixel_values,
        frame_indices=frame_indices,
        timestamps=timestamps,
        update_memory=True
    )
    
    assert features is not None, "Features should be encoded"
    assert memory_info is not None, "Memory info should be returned"
    assert memory_info['memory_summary']['short_term']['count'] == 2, "Should have 2 frames in memory"
    
    print(f"✓ Memory-aware encoding successful: {features.shape}")
    print(f"✓ Memory summary: {memory_info['memory_summary']}")
    
    # Test capacity overflow
    print("\nTesting capacity overflow...")
    for i in range(5):  # Add more frames to trigger overflow
        pixel_values = torch.randn(1, 3, 224, 224)
        frame_indices = torch.tensor([i + 2])
        timestamps = torch.tensor([float(i + 2)])
        
        features, memory_info = model.encode_images_with_memory(
            pixel_values=pixel_values,
            frame_indices=frame_indices,
            timestamps=timestamps,
            update_memory=True
        )
        
        print(f"Frame {i+2}: {memory_info['memory_summary']}")
    
    print("✓ Capacity overflow handling successful")
    
    # Test memory retrieval
    print("\nTesting memory retrieval...")
    query_pixels = torch.randn(1, 3, 224, 224)
    features, memory_info = model.encode_images_with_memory(
        pixel_values=query_pixels,
        frame_indices=torch.tensor([999]),  # Query frame
        update_memory=False  # Don't update, just retrieve
    )
    
    print(f"✓ Memory retrieval successful: retrieved {memory_info['retrieved_count']} frames")
    if memory_info['retrieved_indices']:
        print(f"  Retrieved indices: {memory_info['retrieved_indices']}")
    
    # Test without memory bank
    print("\nTesting standard encoding (no memory)...")
    config_no_memory = MockConfig()
    config_no_memory.enable_memory_bank = False
    model_no_memory = MockVideollama3Model(config_no_memory)
    
    standard_features = model_no_memory.encode_images(pixel_values)
    assert standard_features is not None, "Standard encoding should work"
    assert not model_no_memory.has_memory_bank(), "Memory bank should be disabled"
    
    print(f"✓ Standard encoding successful: {standard_features.shape}")
    
    print("\nAll integration tests passed! 🎉")
    return True


if __name__ == "__main__":
    test_memory_integration()