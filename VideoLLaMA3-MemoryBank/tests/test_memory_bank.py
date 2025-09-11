"""
Unit tests for HierarchicalMemoryBank.
Tests core functionality including memory management, feature adaptation, and retrieval.
"""

import torch
import pytest
import sys
import os

# Add the parent directory to sys.path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from videollama3.model.memory_bank import (
    HierarchicalMemoryBank, 
    MemoryBankConfig, 
    FeatureAdapter
)


class TestFeatureAdapter:
    """Test FeatureAdapter functionality."""
    
    def test_feature_adapter_init(self):
        """Test FeatureAdapter initialization."""
        adapter = FeatureAdapter(input_dim=1152, output_dim=2048)
        assert adapter.adapter is not None
        
    def test_feature_adapter_forward(self):
        """Test FeatureAdapter forward pass."""
        adapter = FeatureAdapter(input_dim=512, output_dim=1024)
        x = torch.randn(2, 100, 512)
        output = adapter(x)
        assert output.shape == (2, 100, 1024)


class TestMemoryBankConfig:
    """Test MemoryBankConfig functionality."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MemoryBankConfig()
        assert config.capacities == [8, 4, 1]
        assert config.reduced_sizes == [144, 64, 36]
        assert config.feature_dim == 1152
        assert config.enable_compression is True
        
    def test_custom_config(self):
        """Test custom configuration values."""
        config = MemoryBankConfig(
            capacities=[10, 5, 2],
            reduced_sizes=[256, 128, 64],
            feature_dim=2048
        )
        assert config.capacities == [10, 5, 2]
        assert config.reduced_sizes == [256, 128, 64]
        assert config.feature_dim == 2048


class TestHierarchicalMemoryBank:
    """Test HierarchicalMemoryBank functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return MemoryBankConfig(
            capacities=[3, 2, 1],
            reduced_sizes=[144, 64, 36],
            feature_dim=1152
        )
    
    @pytest.fixture
    def memory_bank(self, config):
        """Create test memory bank."""
        return HierarchicalMemoryBank(config)
    
    def test_memory_bank_init(self, memory_bank):
        """Test memory bank initialization."""
        assert len(memory_bank.groups) == 3
        assert memory_bank.groups[0]["name"] == "short_term"
        assert memory_bank.groups[1]["name"] == "mid_term"
        assert memory_bank.groups[2]["name"] == "long_term"
        
    def test_meanpool_4d(self, memory_bank):
        """Test mean pooling with 4D tensors."""
        tokens = torch.randn(1, 1152, 12, 12)
        pooled = memory_bank._meanpool(tokens)
        assert pooled.shape == (1, 1152)
        
    def test_meanpool_3d(self, memory_bank):
        """Test mean pooling with 3D tensors."""
        tokens = torch.randn(1, 144, 1152)
        pooled = memory_bank._meanpool(tokens)
        assert pooled.shape == (1, 1152)
        
    def test_reduce_tokens_4d(self, memory_bank):
        """Test token reduction with 4D tensors."""
        tokens = torch.randn(1, 1152, 16, 16)
        reduced = memory_bank._reduce_tokens(tokens, target_size=144)
        expected_h = expected_w = int(144 ** 0.5)  # 12x12 = 144
        assert reduced.shape == (1, 1152, expected_h, expected_w)
        
    def test_reduce_tokens_3d(self, memory_bank):
        """Test token reduction with 3D tensors."""
        tokens = torch.randn(1, 256, 1152)
        reduced = memory_bank._reduce_tokens(tokens, target_size=144)
        assert reduced.shape == (1, 144, 1152)
        
    def test_update_memory_single_frame(self, memory_bank):
        """Test updating memory with a single frame."""
        tokens = torch.randn(1, 1152, 12, 12)  # 144 tokens
        memory_bank.update_memory(tokens, index=0)
        
        assert len(memory_bank.groups[0]["tokens"]) == 1
        assert memory_bank.groups[0]["tokens"][0]["index"] == 0
        
    def test_update_memory_capacity_overflow(self, memory_bank):
        """Test memory updates with capacity overflow."""
        # Fill short-term memory beyond capacity
        for i in range(5):  # Capacity is 3, so this should trigger overflow
            tokens = torch.randn(1, 1152, 12, 12)
            memory_bank.update_memory(tokens, index=i)
        
        # Short-term should be at capacity
        assert len(memory_bank.groups[0]["tokens"]) == 3
        # Mid-term should have received overflowed items
        assert len(memory_bank.groups[1]["tokens"]) >= 1
        
    def test_find_most_similar_frame(self, memory_bank):
        """Test finding most similar frame for eviction."""
        # Add some frames to a group
        group = []
        for i in range(3):
            tokens = torch.randn(1, 1152, 12, 12)
            cls_token = memory_bank._meanpool(tokens)
            group.append({
                "tokens": tokens,
                "index": i,
                "cls_token": cls_token
            })
        
        similar_idx = memory_bank._find_most_similar_frame(group)
        assert 0 <= similar_idx < len(group)
        
    def test_get_memory_summary(self, memory_bank):
        """Test memory summary generation."""
        # Add some frames
        for i in range(2):
            tokens = torch.randn(1, 1152, 12, 12)
            memory_bank.update_memory(tokens, index=i)
            
        summary = memory_bank.get_memory_summary()
        
        assert "short_term" in summary
        assert "mid_term" in summary
        assert "long_term" in summary
        assert summary["short_term"]["count"] == 2
        assert summary["short_term"]["capacity"] == 3
        
    def test_retrieve_memory(self, memory_bank):
        """Test memory retrieval based on query features."""
        # Add some frames
        stored_tokens = []
        for i in range(3):
            tokens = torch.randn(1, 1152, 12, 12)
            stored_tokens.append(tokens)
            memory_bank.update_memory(tokens, index=i)
        
        # Query with one of the stored tokens
        query_features = stored_tokens[0]
        retrieved = memory_bank.retrieve_memory(query_features, top_k=2)
        
        assert len(retrieved) >= 1
        assert all("similarity" in item for item in retrieved)
        assert all("item" in item for item in retrieved)
        
    def test_clear_memory(self, memory_bank):
        """Test memory clearing."""
        # Add some frames
        for i in range(2):
            tokens = torch.randn(1, 1152, 12, 12)
            memory_bank.update_memory(tokens, index=i)
            
        assert memory_bank.get_total_memory_size() == 2
        
        memory_bank.clear_memory()
        assert memory_bank.get_total_memory_size() == 0
        
    def test_export_import_memory_state(self, memory_bank):
        """Test exporting and importing memory state."""
        # Add some frames
        for i in range(2):
            tokens = torch.randn(1, 1152, 12, 12)
            memory_bank.update_memory(tokens, index=i, timestamp=float(i))
            
        # Export state
        state = memory_bank.export_memory_state()
        assert "config" in state
        assert "groups" in state
        
        # Create new memory bank from state
        new_memory_bank = HierarchicalMemoryBank.from_exported_state(state, device="cpu")
        
        # Verify state is preserved
        assert new_memory_bank.get_total_memory_size() == memory_bank.get_total_memory_size()
        
    def test_timestamp_tracking(self, memory_bank):
        """Test timestamp tracking in memory items."""
        tokens = torch.randn(1, 1152, 12, 12)
        timestamp = 1.5
        
        memory_bank.update_memory(tokens, index=0, timestamp=timestamp)
        
        item = memory_bank.groups[0]["tokens"][0]
        assert item["timestamp"] == timestamp


if __name__ == "__main__":
    # Run basic smoke test
    print("Running HierarchicalMemoryBank smoke test...")
    
    # Test configuration
    config = MemoryBankConfig()
    print(f"✓ Config: capacities={config.capacities}, reduced_sizes={config.reduced_sizes}")
    
    # Test memory bank creation
    memory_bank = HierarchicalMemoryBank(config)
    print(f"✓ Memory bank created with {len(memory_bank.groups)} groups")
    
    # Test adding frames
    for i in range(5):
        tokens = torch.randn(1, 1152, 12, 12)
        memory_bank.update_memory(tokens, index=i, timestamp=float(i))
    
    summary = memory_bank.get_memory_summary()
    print(f"✓ Memory summary: {summary}")
    
    # Test retrieval
    query_tokens = torch.randn(1, 1152, 12, 12)
    retrieved = memory_bank.retrieve_memory(query_tokens, top_k=3)
    print(f"✓ Retrieved {len(retrieved)} similar frames")
    
    # Test feature adapter
    adapter = FeatureAdapter(1152, 2048)
    test_features = torch.randn(2, 100, 1152)
    adapted = adapter(test_features)
    print(f"✓ Feature adapter: {test_features.shape} -> {adapted.shape}")
    
    print("All tests passed! ✨")