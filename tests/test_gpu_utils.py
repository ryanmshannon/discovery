"""Tests for multi-GPU utilities in Discovery.

These tests are designed to work on systems with or without GPUs.
GPU-specific tests are marked with @pytest.mark.gpu and will be skipped
if no GPUs are available.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

import discovery as ds
from discovery import gpu_utils


class TestGPUDetection:
    """Tests for GPU detection and device management."""
    
    def test_get_gpu_devices_returns_list(self):
        """Test that get_gpu_devices returns a list."""
        devices = gpu_utils.get_gpu_devices()
        assert isinstance(devices, list)
    
    def test_get_num_gpus_returns_int(self):
        """Test that get_num_gpus returns an integer."""
        num_gpus = gpu_utils.get_num_gpus()
        assert isinstance(num_gpus, int)
        assert num_gpus >= 0
    
    def test_check_multi_gpu_available(self):
        """Test multi-GPU availability check."""
        available, num_gpus = gpu_utils.check_multi_gpu_available()
        assert isinstance(available, bool)
        assert isinstance(num_gpus, int)
        
        # If available, should have > 1 GPU
        if available:
            assert num_gpus > 1
    
    def test_setup_gpu_environment_no_errors(self):
        """Test that setup_gpu_environment runs without errors."""
        # This should not raise an error even without GPUs
        devices = gpu_utils.setup_gpu_environment()
        assert isinstance(devices, list)


class TestDataSharding:
    """Tests for data sharding utilities."""
    
    def test_shard_data_basic(self):
        """Test basic data sharding."""
        data = jnp.arange(8).reshape(8, 1)
        sharded = gpu_utils.shard_data(data, 2)
        
        assert sharded.shape == (2, 4, 1)
        # Check that data is preserved
        assert jnp.array_equal(sharded.reshape(8, 1), data)
    
    def test_shard_data_multidimensional(self):
        """Test sharding of multidimensional arrays."""
        data = jnp.arange(24).reshape(6, 4)
        sharded = gpu_utils.shard_data(data, 3)
        
        assert sharded.shape == (3, 2, 4)
        assert jnp.array_equal(sharded.reshape(6, 4), data)
    
    def test_shard_data_invalid_division(self):
        """Test that sharding raises error for invalid divisions."""
        data = jnp.arange(7).reshape(7, 1)
        
        with pytest.raises(ValueError, match="not evenly divisible"):
            gpu_utils.shard_data(data, 2)
    
    def test_unshard_data_basic(self):
        """Test basic data unsharding."""
        sharded = jnp.arange(8).reshape(2, 4, 1)
        unsharded = gpu_utils.unshard_data(sharded)
        
        assert unsharded.shape == (8, 1)
        assert jnp.array_equal(unsharded, jnp.arange(8).reshape(8, 1))
    
    def test_shard_unshard_roundtrip(self):
        """Test that shard and unshard are inverse operations."""
        data = jnp.arange(12).reshape(12, 1)
        
        sharded = gpu_utils.shard_data(data, 3)
        recovered = gpu_utils.unshard_data(sharded)
        
        assert jnp.array_equal(data, recovered)


class TestOptimalDeviceCount:
    """Tests for optimal device count selection."""
    
    def test_get_optimal_device_count_even_division(self):
        """Test optimal device count with even division."""
        # 12 tasks should optimally use 4 devices if available
        optimal = gpu_utils.get_optimal_device_count(12, max_devices=4)
        assert optimal in [1, 2, 3, 4]
        assert 12 % optimal == 0
    
    def test_get_optimal_device_count_prime_number(self):
        """Test optimal device count with prime number of tasks."""
        # 13 tasks (prime) should use 1 device for even division
        optimal = gpu_utils.get_optimal_device_count(13, max_devices=4)
        # Could be 1 (even division) or up to 4 (no even division possible)
        assert 1 <= optimal <= 4
    
    def test_get_optimal_device_count_more_devices_than_tasks(self):
        """Test when there are more devices than tasks."""
        optimal = gpu_utils.get_optimal_device_count(3, max_devices=10)
        assert optimal <= 3  # Can't use more devices than tasks
    
    def test_get_optimal_device_count_no_gpus(self, monkeypatch):
        """Test behavior when no GPUs are available."""
        # Mock get_num_gpus to return 0
        monkeypatch.setattr(gpu_utils, "get_num_gpus", lambda: 0)
        
        optimal = gpu_utils.get_optimal_device_count(10)
        assert optimal == 1


@pytest.mark.gpu
class TestGPUSpecificFeatures:
    """Tests that require actual GPU hardware.
    
    These tests will be skipped if no GPUs are available.
    """
    
    @pytest.fixture(autouse=True)
    def check_gpu_available(self):
        """Skip tests if no GPU is available."""
        if gpu_utils.get_num_gpus() == 0:
            pytest.skip("No GPU available")
    
    def test_distribute_to_devices(self):
        """Test distributing data to GPU devices."""
        devices = gpu_utils.get_gpu_devices()
        data = jnp.array([1, 2, 3, 4])
        
        distributed = gpu_utils.distribute_to_devices(data, devices[:1])
        
        assert len(distributed) == 1
        assert jnp.array_equal(distributed[0], data)
    
    def test_pmap_wrapper_basic(self):
        """Test basic pmap wrapper functionality."""
        devices = gpu_utils.get_gpu_devices()
        
        if len(devices) < 2:
            pytest.skip("Need at least 2 GPUs")
        
        def simple_func(x):
            return x * 2
        
        pmapped_func = gpu_utils.pmap_wrapper(simple_func, devices=devices[:2])
        
        # Create data for 2 devices
        data = jnp.array([[1, 2], [3, 4]])
        result = pmapped_func(data)
        
        expected = data * 2
        assert jnp.array_equal(result, expected)


class TestGPUUtilsIntegration:
    """Integration tests for GPU utilities with Discovery components."""
    
    def test_gpu_utils_import(self):
        """Test that gpu_utils can be imported from discovery."""
        assert hasattr(ds, 'gpu_utils')
        assert hasattr(ds.gpu_utils, 'get_gpu_devices')
        assert hasattr(ds.gpu_utils, 'get_num_gpus')
    
    def test_setup_gpu_environment_with_specific_gpus(self, monkeypatch):
        """Test setup with specific GPU IDs."""
        # Mock to avoid actually changing CUDA_VISIBLE_DEVICES
        original_environ = {}
        
        def mock_get_gpu_devices():
            # Return a mock device list
            return []
        
        monkeypatch.setattr(gpu_utils, "get_gpu_devices", mock_get_gpu_devices)
        
        # Should not raise error
        devices = gpu_utils.setup_gpu_environment(gpu_ids=[0, 1])
        assert isinstance(devices, list)
    
    def test_shard_data_with_jax_arrays(self):
        """Test sharding with JAX arrays (not numpy)."""
        data = jax.random.normal(jax.random.PRNGKey(0), (10, 5))
        sharded = gpu_utils.shard_data(data, 2)
        
        assert sharded.shape == (2, 5, 5)
        assert isinstance(sharded, jax.Array)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_shard_data_single_device(self):
        """Test sharding with a single device (no-op case)."""
        data = jnp.arange(10).reshape(10, 1)
        sharded = gpu_utils.shard_data(data, 1)
        
        assert sharded.shape == (1, 10, 1)
    
    def test_get_optimal_device_count_single_task(self):
        """Test optimal device count for single task."""
        optimal = gpu_utils.get_optimal_device_count(1, max_devices=4)
        assert optimal == 1
    
    def test_setup_gpu_environment_empty_gpu_list(self):
        """Test setup with empty GPU ID list."""
        # Should use all available GPUs
        devices = gpu_utils.setup_gpu_environment(gpu_ids=[])
        assert isinstance(devices, list)
    
    def test_unshard_data_single_device(self):
        """Test unsharding from single device."""
        sharded = jnp.arange(10).reshape(1, 10, 1)
        unsharded = gpu_utils.unshard_data(sharded)
        
        assert unsharded.shape == (10, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
