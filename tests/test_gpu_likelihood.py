"""Tests for multi-GPU likelihood functionality in Discovery.

These tests verify that the gpu_logL method works correctly for
GlobalLikelihood objects.  Multi-device behaviour is verified using
simulated CPU devices (enabled by setting XLA_FLAGS in conftest.py).
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

import discovery as ds
from discovery import gpu_utils, matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_psl(seed=0, n_toa=50, n_fourier=10):
    """Create a minimal PulsarLikelihood with a variable-P WoodburyKernel."""
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n_toa)
    noise = matrix.NoiseMatrix1D_novar(np.ones(n_toa))
    F = rng.standard_normal((n_toa, n_fourier))

    param_name = f'log_amp_{seed}'

    def getN(params):
        return jnp.exp(params[param_name]) * jnp.ones(n_fourier)
    getN.params = [param_name]

    prior = matrix.NoiseMatrix1D_var(getN)
    kern = matrix.WoodburyKernel(noise, F, prior)
    return ds.PulsarLikelihood([y, kern])


def _make_test_globalgp(n_psr, n_toa=50, n_gp=5):
    """Create a minimal GlobalVariableGP covering *n_psr* pulsars."""
    rng = np.random.default_rng(99)
    Fs = [rng.standard_normal((n_toa, n_gp)) for _ in range(n_psr)]

    def getPhi(params):
        return jnp.exp(params['gp_log_amp']) * jnp.ones(n_psr * n_gp)
    getPhi.params = ['gp_log_amp']

    phi = matrix.NoiseMatrix1D_var(getPhi)
    return matrix.GlobalVariableGP(phi, Fs)


def _multi_cpu_devices(n=2):
    """Return *n* CPU devices; skip the test if fewer are available."""
    devices = jax.devices('cpu')
    if len(devices) < n:
        pytest.skip(
            f"Need at least {n} CPU devices.  "
            f"Set XLA_FLAGS=--xla_force_host_platform_device_count={n} "
            "before importing JAX."
        )
    return devices[:n]


# ---------------------------------------------------------------------------
# Structural / API tests
# ---------------------------------------------------------------------------

class TestGPULikelihoodBasic:
    """Basic tests for gpu_logL that work without actual GPUs."""

    def test_gpu_logl_method_exists(self):
        """GlobalLikelihood exposes a gpu_logL method."""
        assert hasattr(ds.GlobalLikelihood, 'gpu_logL')

    def test_gpu_logl_returns_callable(self):
        """gpu_logL returns a callable with a .devices attribute."""
        devices = _multi_cpu_devices(2)
        psls = [_make_test_psl(i) for i in range(4)]
        gbl = ds.GlobalLikelihood(psls)
        fn = gbl.gpu_logL(devices=devices, use_pmap=True)
        assert callable(fn)
        assert hasattr(fn, 'devices')
        assert list(fn.devices) == list(devices)

    def test_gpu_logl_raises_when_more_devices_than_pulsars(self):
        """gpu_logL raises ValueError when #devices > #pulsars."""
        devices = _multi_cpu_devices(2)
        psls = [_make_test_psl(0)]  # only 1 pulsar
        gbl = ds.GlobalLikelihood(psls)
        with pytest.raises(ValueError, match="less than number of devices"):
            gbl.gpu_logL(devices=devices, use_pmap=True)


# ---------------------------------------------------------------------------
# Multi-device correctness tests (no global GP)
# ---------------------------------------------------------------------------

class TestMultiDeviceNoGlobalGP:
    """Verify gpu_logL correctness for the simple (no globalgp) case."""

    @pytest.fixture
    def setup(self):
        devices = _multi_cpu_devices(2)
        n_psr = 4
        psls = [_make_test_psl(i) for i in range(n_psr)]
        gbl = ds.GlobalLikelihood(psls)
        params = {f'log_amp_{i}': jnp.array(float(i) * 0.2 - 0.3) for i in range(n_psr)}
        return gbl, params, devices

    def test_pmap_result_matches_single_device(self, setup):
        """Multi-device (use_pmap=True) result matches single-device logL."""
        gbl, params, devices = setup
        single = float(gbl.logL(params))
        fn = gbl.gpu_logL(devices=devices, use_pmap=True)
        multi = float(fn(params))
        assert np.isclose(single, multi, rtol=1e-5), (
            f"single={single}, multi={multi}"
        )

    def test_sequential_fallback_matches_single_device(self, setup):
        """Sequential fallback (use_pmap=False) result matches single-device logL."""
        gbl, params, devices = setup
        single = float(gbl.logL(params))
        fn = gbl.gpu_logL(devices=devices, use_pmap=False)
        seq = float(fn(params))
        assert np.isclose(single, seq, rtol=1e-5), (
            f"single={single}, sequential={seq}"
        )

    def test_device_placement_of_closure_arrays(self, setup):
        """Arrays captured in each device group's closures reside on that device."""
        gbl, _, devices = setup
        fn = gbl.gpu_logL(devices=devices, use_pmap=True)

        # Extract logl_groups and device_list from the closure.
        logl_groups = None
        device_list = None
        for cell in fn.__closure__:
            try:
                v = cell.cell_contents
            except ValueError:
                continue
            if isinstance(v, list) and v and isinstance(v[0], jax.Device):
                device_list = v
            elif (isinstance(v, list) and v
                    and isinstance(v[0], list) and v[0]
                    and callable(v[0][0])):
                logl_groups = v

        assert logl_groups is not None, "Could not find logl_groups in closure"
        assert device_list is not None, "Could not find device_list in closure"

        for group, expected_device in zip(logl_groups, device_list):
            for logl_fn in group:
                if not logl_fn.__closure__:
                    continue
                for cell in logl_fn.__closure__:
                    try:
                        v = cell.cell_contents
                    except ValueError:
                        continue
                    if isinstance(v, jax.Array):
                        assert v.device == expected_device, (
                            f"Expected array on {expected_device}, "
                            f"got {v.device}"
                        )

    def test_result_on_primary_device(self, setup):
        """The scalar result of gpu_logL is placed on the primary device."""
        gbl, params, devices = setup
        fn = gbl.gpu_logL(devices=devices, use_pmap=True)
        result = fn(params)
        assert isinstance(result, jax.Array)
        assert result.device == devices[0]

    def test_uneven_pulsar_distribution(self):
        """3 pulsars across 2 devices (uneven) gives correct results."""
        devices = _multi_cpu_devices(2)
        n_psr = 3
        psls = [_make_test_psl(i) for i in range(n_psr)]
        gbl = ds.GlobalLikelihood(psls)
        params = {f'log_amp_{i}': jnp.array(0.1 * i) for i in range(n_psr)}

        single = float(gbl.logL(params))
        fn = gbl.gpu_logL(devices=devices, use_pmap=True)
        multi = float(fn(params))
        assert np.isclose(single, multi, rtol=1e-5), (
            f"single={single}, multi={multi}"
        )

    def test_four_devices(self):
        """Distributing 4 pulsars across 4 devices gives correct results."""
        devices = _multi_cpu_devices(4)
        n_psr = 4
        psls = [_make_test_psl(i) for i in range(n_psr)]
        gbl = ds.GlobalLikelihood(psls)
        params = {f'log_amp_{i}': jnp.array(-0.5 + 0.25 * i) for i in range(n_psr)}

        single = float(gbl.logL(params))
        fn = gbl.gpu_logL(devices=devices, use_pmap=True)
        multi = float(fn(params))
        assert np.isclose(single, multi, rtol=1e-5), (
            f"single={single}, multi={multi}"
        )


# ---------------------------------------------------------------------------
# Multi-device correctness tests (with global GP)
# ---------------------------------------------------------------------------

class TestMultiDeviceWithGlobalGP:
    """Verify gpu_logL correctness for the globalgp case."""

    @pytest.fixture
    def setup(self):
        devices = _multi_cpu_devices(2)
        n_psr = 4
        psls = [_make_test_psl(i) for i in range(n_psr)]
        globalgp = _make_test_globalgp(n_psr)
        gbl = ds.GlobalLikelihood(psls, globalgp=globalgp)
        params = {f'log_amp_{i}': jnp.array(float(i) * 0.2 - 0.3)
                  for i in range(n_psr)}
        params['gp_log_amp'] = jnp.array(-1.0)
        return gbl, params, devices

    def test_pmap_result_matches_single_device(self, setup):
        """Multi-device result matches single-device for globalgp model."""
        gbl, params, devices = setup
        single = float(gbl.logL(params))
        fn = gbl.gpu_logL(devices=devices, use_pmap=True)
        multi = float(fn(params))
        assert np.isclose(single, multi, rtol=1e-5), (
            f"single={single}, multi={multi}"
        )

    def test_sequential_fallback_matches_single_device(self, setup):
        """Sequential fallback matches single-device for globalgp model."""
        gbl, params, devices = setup
        single = float(gbl.logL(params))
        fn = gbl.gpu_logL(devices=devices, use_pmap=False)
        seq = float(fn(params))
        assert np.isclose(single, seq, rtol=1e-5), (
            f"single={single}, sequential={seq}"
        )

    def test_device_placement_of_kterm_closure_arrays(self, setup):
        """Arrays captured in each kterm group's closures reside on that device."""
        gbl, _, devices = setup
        fn = gbl.gpu_logL(devices=devices, use_pmap=True)

        kterm_groups = None
        device_list = None
        for cell in fn.__closure__:
            try:
                v = cell.cell_contents
            except ValueError:
                continue
            if isinstance(v, list) and v and isinstance(v[0], jax.Device):
                device_list = v
            elif (isinstance(v, list) and v
                    and isinstance(v[0], list) and v[0]
                    and callable(v[0][0])):
                kterm_groups = v

        assert kterm_groups is not None, "Could not find kterm_groups in closure"
        assert device_list is not None, "Could not find device_list in closure"

        for group, expected_device in zip(kterm_groups, device_list):
            for kterm_fn in group:
                if not kterm_fn.__closure__:
                    continue
                for cell in kterm_fn.__closure__:
                    try:
                        v = cell.cell_contents
                    except ValueError:
                        continue
                    if isinstance(v, jax.Array):
                        assert v.device == expected_device, (
                            f"Expected array on {expected_device}, "
                            f"got {v.device}"
                        )

    def test_uneven_pulsar_distribution(self):
        """3 pulsars across 2 devices (uneven) gives correct results."""
        devices = _multi_cpu_devices(2)
        n_psr = 3
        psls = [_make_test_psl(i) for i in range(n_psr)]
        globalgp = _make_test_globalgp(n_psr)
        gbl = ds.GlobalLikelihood(psls, globalgp=globalgp)
        params = {f'log_amp_{i}': jnp.array(0.1 * i) for i in range(n_psr)}
        params['gp_log_amp'] = jnp.array(-0.5)

        single = float(gbl.logL(params))
        fn = gbl.gpu_logL(devices=devices, use_pmap=True)
        multi = float(fn(params))
        assert np.isclose(single, multi, rtol=1e-5), (
            f"single={single}, multi={multi}"
        )


# ---------------------------------------------------------------------------
# Real GPU hardware tests (skipped when no GPU is available)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestGPULikelihoodWithHardware:
    """Tests that require actual GPU hardware."""

    @pytest.fixture(autouse=True)
    def check_multi_gpu_available(self):
        """Skip tests if multiple GPUs are not available."""
        available, num_gpus = gpu_utils.check_multi_gpu_available()
        if not available:
            pytest.skip(f"Need multiple GPUs, only {num_gpus} available")


# ---------------------------------------------------------------------------
# Performance tests (opt-in only)
# ---------------------------------------------------------------------------

class TestGPULikelihoodPerformance:
    """Performance-related tests (run manually with -m slow)."""

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_gpu_logl_speedup(self):
        pytest.skip("Performance test - run manually")

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_gpu_logl_scaling(self):
        pytest.skip("Scaling test - run manually")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
