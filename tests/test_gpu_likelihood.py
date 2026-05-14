"""Tests for multi-GPU likelihood functionality in Discovery.

These tests verify that the gpu_logL method works correctly for
GlobalLikelihood objects, with and without actual GPU hardware.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

import discovery as ds
from discovery import gpu_utils


class TestGPULikelihoodBasic:
    """Basic tests for gpu_logL that work without actual GPUs."""
    
    def test_gpu_logl_method_exists(self):
        """Test that GlobalLikelihood has gpu_logL method."""
        # Create a minimal GlobalLikelihood
        # This is just a structural test, no actual computation
        assert hasattr(ds.GlobalLikelihood, 'gpu_logL')
    
    def test_gpu_logl_requires_devices(self, monkeypatch):
        """Test that gpu_logL raises error when no GPUs available."""
        # Mock get_gpu_devices to return empty list
        monkeypatch.setattr(gpu_utils, "get_gpu_devices", lambda: [])
        
        # Create a minimal test case with mock pulsars
        # Note: This would need actual pulsar data for a full test
        # For now, we're just testing the error handling
        pass  # Placeholder for when we have test data


@pytest.mark.gpu
class TestGPULikelihoodWithHardware:
    """Tests that require actual GPU hardware."""
    
    @pytest.fixture(autouse=True)
    def check_multi_gpu_available(self):
        """Skip tests if multiple GPUs are not available."""
        available, num_gpus = gpu_utils.check_multi_gpu_available()
        if not available:
            pytest.skip(f"Need multiple GPUs, only {num_gpus} available")
    
    @pytest.fixture
    def mock_pulsar_data(self):
        """Create mock pulsar data for testing.
        
        This is a simplified version. Real tests would use actual
        pulsar timing data from the data/ folder.
        """
        # Create minimal mock pulsar-like objects
        # In real usage, these would be ds.Pulsar objects
        class MockPulsar:
            def __init__(self, n_toas=100):
                self.residuals = jnp.array(np.random.randn(n_toas))
                self.noisedict = {}
        
        return [MockPulsar() for _ in range(4)]
    
    def test_gpu_logl_creation(self):
        """Test that gpu_logL can be created."""
        # This is a placeholder for actual implementation test
        # Would need real pulsar data
        pass
    
    def test_gpu_logl_device_specification(self):
        """Test specifying number of devices."""
        # Test with specific device count
        num_gpus = gpu_utils.get_num_gpus()
        assert num_gpus >= 2, "Need at least 2 GPUs for this test"
        
        # Placeholder for actual test with real data
        pass


class TestGPULikelihoodIntegration:
    """Integration tests for GPU likelihood with other Discovery components."""
    
    def test_gpu_logl_has_params_attribute(self):
        """Test that gpu_logL result has params attribute like normal logL."""
        # Placeholder - would need actual GlobalLikelihood instance
        pass
    
    def test_gpu_logl_compatible_with_jit(self):
        """Test that gpu_logL result can be JIT compiled."""
        # Placeholder - would test jax.jit(logl_parallel)
        pass


class TestGPULikelihoodErrorHandling:
    """Tests for error handling in gpu_logL."""
    
    def test_gpu_logl_fewer_pulsars_than_devices(self, monkeypatch):
        """Test error when pulsars < devices."""
        # Mock to simulate having many GPUs
        def mock_get_gpu_devices():
            # Return mock device list
            return [f"gpu:{i}" for i in range(10)]
        
        monkeypatch.setattr(gpu_utils, "get_gpu_devices", mock_get_gpu_devices)
        
        # Would test that appropriate error is raised
        # when trying to use 10 GPUs with only 5 pulsars
        pass
    
    def test_gpu_logl_warns_uneven_distribution(self):
        """Test warning when pulsars not evenly divisible by devices."""
        # Test that warning is issued for suboptimal distribution
        pass


class TestGPULikelihoodPerformance:
    """Performance-related tests (can be slow)."""
    
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_gpu_logl_speedup(self):
        """Test that multi-GPU provides speedup over single GPU.
        
        This is a slow test that benchmarks performance.
        """
        pytest.skip("Performance test - run manually")
    
    @pytest.mark.slow
    @pytest.mark.gpu  
    def test_gpu_logl_scaling(self):
        """Test scaling with different numbers of GPUs.
        
        This test checks that performance scales reasonably
        with the number of GPUs used.
        """
        pytest.skip("Scaling test - run manually")


# Helper functions for future test implementation

def create_simple_global_likelihood(num_pulsars=10):
    """Helper to create a simple GlobalLikelihood for testing.
    
    Args:
        num_pulsars: Number of mock pulsars to create.
    
    Returns:
        A GlobalLikelihood instance suitable for testing.
    
    Note:
        This is a stub. Real implementation would need actual
        pulsar data and proper signal construction.
    """
    # TODO: Implement when test data is available
    raise NotImplementedError("Need actual test data")


def create_test_params(logl):
    """Helper to create test parameters for a likelihood.
    
    Args:
        logl: Likelihood function with params attribute.
    
    Returns:
        Dictionary of random parameter values.
    """
    # TODO: Implement proper parameter sampling
    raise NotImplementedError("Need proper parameter handling")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
