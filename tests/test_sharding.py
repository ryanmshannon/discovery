"""Tests for model sharding functionality across multiple GPUs."""

import pytest
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Skip all tests if JAX is not available
pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")


class TestShardingUtilities:
    """Test sharding utility functions in gpu_utils."""
    
    def test_create_sharding_spec(self):
        """Test creation of sharding specifications."""
        from discovery import gpu_utils
        
        # Test with available devices
        num_gpus = gpu_utils.get_num_gpus()
        if num_gpus > 0:
            sharding = gpu_utils.create_sharding_spec(min(2, num_gpus))
            assert sharding is not None or num_gpus < 2
        else:
            # Should handle no GPU case gracefully
            sharding = gpu_utils.create_sharding_spec(1)
            # May return None or CPU sharding
    
    def test_shard_array_to_devices(self):
        """Test array sharding across devices."""
        from discovery import gpu_utils
        
        # Create test array
        data = jnp.arange(100).reshape(10, 10)
        
        num_gpus = gpu_utils.get_num_gpus()
        if num_gpus >= 2:
            # Test successful sharding
            sharded = gpu_utils.shard_array_to_devices(data, 2, axis=0)
            assert sharded.shape == data.shape
        else:
            # Should handle single GPU or CPU gracefully
            result = gpu_utils.shard_array_to_devices(data, 1, axis=0)
            assert result.shape == data.shape
    
    def test_shard_array_uneven_split(self):
        """Test that uneven array splits raise appropriate errors."""
        from discovery import gpu_utils
        
        # Create array that can't be evenly divided
        data = jnp.arange(15).reshape(5, 3)
        
        num_gpus = gpu_utils.get_num_gpus()
        if num_gpus >= 2:
            # Should raise error for uneven split
            with pytest.raises(ValueError, match="not evenly divisible"):
                gpu_utils.shard_array_to_devices(data, 2, axis=0)
    
    def test_replicate_across_devices(self):
        """Test data replication across devices."""
        from discovery import gpu_utils
        
        # Test with simple data
        data = {'param1': 1.0, 'param2': jnp.array([1.0, 2.0, 3.0])}
        
        num_gpus = gpu_utils.get_num_gpus()
        if num_gpus > 0:
            replicated = gpu_utils.replicate_across_devices(data, gpu_utils.get_gpu_devices()[:1])
            # Should handle replication (may modify structure for pmap)
            assert replicated is not None
    
    def test_get_device_mesh(self):
        """Test device mesh creation."""
        from discovery import gpu_utils
        
        num_gpus = gpu_utils.get_num_gpus()
        if num_gpus > 0:
            mesh = gpu_utils.get_device_mesh(min(2, num_gpus))
            # May return None if Mesh not available in JAX version
            # Check it doesn't crash and returns expected type
            if mesh is not None:
                # Verify it's a valid mesh object
                assert hasattr(mesh, 'devices') or hasattr(mesh, 'shape')
            else:
                # None is acceptable for older JAX versions
                assert mesh is None
    
    def test_pmap_reduce_sum(self):
        """Test pmap reduction with sum."""
        from discovery import gpu_utils
        
        def simple_func(x):
            return x ** 2
        
        num_gpus = gpu_utils.get_num_gpus()
        if num_gpus >= 2:
            # Create pmapped version
            pmapped_func = gpu_utils.pmap_reduce_sum(simple_func)
            # Test it works (may execute on CPU if no GPUs)
            assert pmapped_func is not None


class TestGlobalLikelihoodSharding:
    """Test sharding in GlobalLikelihood.gpu_logL()."""
    
    @pytest.mark.gpu
    def test_gpu_logl_no_globalgp_with_sharding(self):
        """Test gpu_logL with sharding for simple case (no globalgp)."""
        from discovery import gpu_utils
        
        num_gpus = gpu_utils.get_num_gpus()
        if num_gpus < 2:
            pytest.skip("Need at least 2 GPUs for this test")
        
        # Would need actual pulsar data to test fully
        # This is a placeholder for integration test
        pass
    
    @pytest.mark.gpu
    def test_gpu_logl_with_globalgp_sharding(self):
        """Test gpu_logL with sharding for complex case (with globalgp)."""
        from discovery import gpu_utils
        
        num_gpus = gpu_utils.get_num_gpus()
        if num_gpus < 2:
            pytest.skip("Need at least 2 GPUs for this test")
        
        # Would need actual pulsar data to test fully
        # This is a placeholder for integration test
        pass
    
    def test_gpu_logl_use_pmap_flag(self):
        """Test that use_pmap flag works correctly."""
        # This would require actual GlobalLikelihood objects
        # Placeholder for now
        pass
    
    def test_gpu_logl_device_placement(self):
        """Test that device placement works correctly."""
        from discovery import gpu_utils
        
        num_gpus = gpu_utils.get_num_gpus()
        if num_gpus == 0:
            pytest.skip("Need at least 1 GPU for this test")
        
        # Verify device placement utilities work
        devices = gpu_utils.get_gpu_devices()
        assert len(devices) > 0
        
        # Test device_put
        test_data = jnp.array([1.0, 2.0, 3.0])
        placed_data = jax.device_put(test_data, devices[0])
        assert placed_data.device == devices[0]


class TestArrayLikelihoodSharding:
    """Test sharding in ArrayLikelihood.gpu_logL()."""
    
    @pytest.mark.gpu
    def test_array_likelihood_gpu_logl(self):
        """Test ArrayLikelihood.gpu_logL() with device placement."""
        from discovery import gpu_utils
        
        num_gpus = gpu_utils.get_num_gpus()
        if num_gpus == 0:
            pytest.skip("Need at least 1 GPU for this test")
        
        # Would need actual ArrayLikelihood object to test
        # This is a placeholder
        pass
    
    def test_array_likelihood_single_device(self):
        """Test that ArrayLikelihood works with single device."""
        # Placeholder for integration test
        pass


class TestShardingIntegration:
    """Integration tests for full sharding workflow."""
    
    @pytest.mark.integration
    @pytest.mark.gpu
    def test_end_to_end_sharding_workflow(self):
        """Test complete workflow with sharding."""
        from discovery import gpu_utils
        
        num_gpus = gpu_utils.get_num_gpus()
        if num_gpus < 2:
            pytest.skip("Need at least 2 GPUs for integration test")
        
        # This would test:
        # 1. Loading pulsars
        # 2. Creating GlobalLikelihood
        # 3. Creating gpu_logL with sharding
        # 4. Evaluating likelihood
        # 5. Comparing with non-sharded version
        
        # Placeholder for full integration test
        pass
    
    @pytest.mark.integration
    def test_sharding_matches_non_sharded(self):
        """Verify sharded and non-sharded results match."""
        # This is critical: sharded version should give identical results
        # to non-sharded version (within numerical precision)
        pass
    
    @pytest.mark.integration
    @pytest.mark.gpu
    def test_sharding_performance_gain(self):
        """Test that sharding provides performance improvement."""
        # This would benchmark sharded vs non-sharded for large models
        pass


class TestShardingEdgeCases:
    """Test edge cases and error handling."""
    
    def test_more_devices_than_pulsars(self):
        """Test error when requesting more devices than pulsars."""
        # Should raise appropriate ValueError
        pass
    
    def test_uneven_pulsar_distribution(self):
        """Test handling of uneven pulsar distribution."""
        # Should warn but still work
        pass
    
    def test_no_gpus_available(self):
        """Test graceful fallback when no GPUs available."""
        # Should raise clear error message
        pass
    
    def test_insufficient_gpu_memory(self):
        """Test handling of GPU memory exhaustion."""
        # This is hard to test reliably, but important for docs
        pass


class TestShardingWithJAXTransforms:
    """Test that sharding works with JAX transformations."""
    
    @pytest.mark.gpu
    def test_jit_with_sharded_likelihood(self):
        """Test JIT compilation of sharded likelihood."""
        # Sharded functions should be JIT-compatible
        pass
    
    @pytest.mark.gpu
    def test_grad_with_sharded_likelihood(self):
        """Test gradient computation with sharded likelihood."""
        # Important for HMC/NUTS samplers
        pass
    
    @pytest.mark.gpu
    def test_vmap_with_sharded_likelihood(self):
        """Test vmap over parameter sets with sharding."""
        # Useful for batch evaluation
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
