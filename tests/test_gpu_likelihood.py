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
    """Create a minimal PulsarLikelihood with a variable-P WoodburyKernel.

    The prior amplitude parameter is named ``log_amp_{seed}`` so that
    multiple pulsars created with different seeds each have a unique
    parameter name.  When constructing parameter dictionaries for tests,
    use ``{f'log_amp_{i}': value}`` for pulsars created with ``seed=i``.

    Args:
        seed: Random seed used to generate residuals and the Fourier matrix.
              Also determines the prior parameter name (``log_amp_{seed}``).
        n_toa: Number of simulated timing residuals (observations).
        n_fourier: Number of Fourier basis functions for the red-noise GP.
    """
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


def _make_test_psl_with_prealloc_freq(seed=0, n_toa=50, n_fourier=10):
    """Create a PulsarLikelihood whose prior captures a pre-allocated JAX array.

    This mimics the behaviour of :func:`~discovery.signals.makegp_fourier`,
    which pre-allocates frequency arrays as JAX arrays at model-setup time::

        f, df = matrix.jnparray(f), matrix.jnparray(df)
        def priorfunc(params):
            return prior(f, df, ...)

    Those arrays land on the default device (GPU:0) when the model is built.
    ``gpu_logL`` must transfer them to the assigned device for each pulsar so
    that evaluation does not trigger cross-device operations.

    Args:
        seed: Random seed; also determines the prior parameter name
              (``log_amp_{seed}``).
        n_toa: Number of simulated timing residuals.
        n_fourier: Number of Fourier basis functions.
    """
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n_toa)
    noise = matrix.NoiseMatrix1D_novar(np.ones(n_toa))
    F = rng.standard_normal((n_toa, n_fourier))

    param_name = f'log_amp_{seed}'

    # Pre-allocate a frequency array on the *current* default device,
    # exactly as makegp_fourier does with jnparray(f).
    freqs = jnp.array(np.linspace(1.0 / 100, n_fourier / 100, n_fourier))

    def getN(params):
        # Uses the pre-allocated `freqs` from the outer scope —
        # this is the array that must be transferred to each device.
        return jnp.exp(params[param_name]) * freqs ** (-4.0 / 3.0)
    getN.params = [param_name]

    prior = matrix.NoiseMatrix1D_var(getN)
    kern = matrix.WoodburyKernel(noise, F, prior)
    return ds.PulsarLikelihood([y, kern])


def _all_closure_arrays(fn, _visited=None):
    """Recursively collect all jax.Array objects found in *fn*'s closure tree.

    Returns a flat list of :class:`jax.Array` objects found at any depth in
    the closure hierarchy of *fn*.
    """
    if _visited is None:
        _visited = set()
    fn_id = id(fn)
    if fn_id in _visited:
        return []
    _visited.add(fn_id)

    results = []
    if not (hasattr(fn, '__closure__') and fn.__closure__):
        return results

    for cell in fn.__closure__:
        try:
            v = cell.cell_contents
        except ValueError:
            continue
        if isinstance(v, jax.Array):
            results.append(v)
        elif callable(v) and hasattr(v, '__closure__') and v.__closure__:
            results.extend(_all_closure_arrays(v, _visited))
        elif isinstance(v, (list, tuple)):
            for item in v:
                if isinstance(item, jax.Array):
                    results.append(item)
                elif callable(item) and hasattr(item, '__closure__') and item.__closure__:
                    results.extend(_all_closure_arrays(item, _visited))
        elif isinstance(v, dict):
            for val in v.values():
                if isinstance(val, jax.Array):
                    results.append(val)
                elif callable(val) and hasattr(val, '__closure__') and val.__closure__:
                    results.extend(_all_closure_arrays(val, _visited))
    return results


def _make_test_globalgp(n_psr, n_toa=50, n_gp=5):
    """Create a minimal GlobalVariableGP covering *n_psr* pulsars.

    Uses a fixed random seed (99) for reproducibility.  The global prior
    has a single scalar amplitude parameter named ``gp_log_amp``.

    Args:
        n_psr: Number of pulsars (one Fourier matrix is created per pulsar).
        n_toa: Number of timing residuals per pulsar (default 50).
        n_gp: Number of Fourier components in the global GP (default 5).
    """
    rng = np.random.default_rng(99)
    Fs = [rng.standard_normal((n_toa, n_gp)) for _ in range(n_psr)]

    def getPhi(params):
        return jnp.exp(params['gp_log_amp']) * jnp.ones(n_psr * n_gp)
    getPhi.params = ['gp_log_amp']

    phi = matrix.NoiseMatrix1D_var(getPhi)
    return matrix.GlobalVariableGP(phi, Fs)


def _multi_cpu_devices(n=2):
    """Return *n* CPU devices; skip the calling test if fewer are available.

    During the normal test suite ``conftest.py`` sets
    ``XLA_FLAGS=--xla_force_host_platform_device_count=4`` before JAX is
    imported, so 4 CPU devices are always available.  This helper will
    skip a test only when running outside the test suite without that flag.
    """
    devices = jax.devices('cpu')
    if len(devices) < n:
        pytest.skip(
            f"Need at least {n} CPU devices.  "
            "conftest.py sets XLA_FLAGS to provide 4 simulated CPU devices; "
            "ensure JAX is imported after that environment variable is set."
        )
    return devices[:n]


def _extract_closure_groups(fn):
    """Extract ``(groups, device_list)`` from a ``gpu_logL`` closure.

    The ``loglike`` function returned by ``gpu_logL`` closes over a list of
    :class:`jax.Device` objects and a list-of-lists of callable kernel
    closures.  This helper locates those two objects by inspecting the
    Python closure cells of *fn* and returns them as a 2-tuple.

    Returns:
        ``(groups, device_list)`` where *groups* is a list of lists of
        callable closures and *device_list* is a list of :class:`jax.Device`.
        Either element may be ``None`` if it could not be found.
    """
    groups = None
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
            groups = v
    return groups, device_list


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

        logl_groups, device_list = _extract_closure_groups(fn)

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

    def test_fresh_closures_not_cached_logl(self):
        """gpu_logL builds fresh per-device closures, not cached psl.logL.

        Accessing psl.logL before calling gpu_logL caches arrays on the
        primary device.  The use_pmap=True path must build new closures via
        make_kernelproduct inside jax.default_device(device) so that each
        device's captured arrays land on the correct device.
        """
        devices = _multi_cpu_devices(2)
        n_psr = 4
        psls = [_make_test_psl(i) for i in range(n_psr)]
        gbl = ds.GlobalLikelihood(psls)
        params = {f'log_amp_{i}': jnp.array(float(i) * 0.1) for i in range(n_psr)}

        # Force psl.logL to be cached on device[0] (primary).
        _ = [psl.logL for psl in psls]

        # gpu_logL must still produce the correct answer even though psl.logL
        # has been pre-cached, because it rebuilds closures from scratch.
        single = float(gbl.logL(params))
        fn = gbl.gpu_logL(devices=devices, use_pmap=True)
        multi = float(fn(params))
        assert np.isclose(single, multi, rtol=1e-5), (
            f"single={single}, multi={multi}"
        )

        # Verify the closures' arrays are on their assigned devices.
        kterm_groups, device_list = _extract_closure_groups(fn)
        assert kterm_groups is not None
        assert device_list is not None
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
                            f"Expected array on {expected_device}, got {v.device}"
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

        kterm_groups, device_list = _extract_closure_groups(fn)

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
# Tests for pre-allocated JAX arrays in prior functions (the core bug)
# ---------------------------------------------------------------------------

class TestPreallocatedArrayTransfer:
    """Verify that pre-allocated JAX arrays in prior closures are transferred.

    Signal functions such as ``makegp_fourier`` call ``jnparray`` on frequency
    arrays at model-setup time, placing them on the default device (GPU:0).
    Those arrays end up nested inside the ``getN`` / ``priorfunc`` callables
    that are captured by the per-device kernel closures built by ``gpu_logL``.

    ``gpu_logL`` must call ``put_closure_arrays_on_device`` on every kernel
    closure after construction so that these deeply-nested arrays are also
    moved to the assigned device, preventing cross-device operations at
    evaluation time.
    """

    @pytest.fixture
    def setup(self):
        devices = _multi_cpu_devices(2)
        n_psr = 4
        # Use the helper whose prior captures a pre-allocated JAX frequency array.
        psls = [_make_test_psl_with_prealloc_freq(i) for i in range(n_psr)]
        gbl = ds.GlobalLikelihood(psls)
        params = {f'log_amp_{i}': jnp.array(float(i) * 0.2 - 0.3) for i in range(n_psr)}
        return gbl, params, devices

    def test_pmap_result_matches_single_device(self, setup):
        """Multi-device result matches single-device even with pre-allocated freq arrays."""
        gbl, params, devices = setup
        single = float(gbl.logL(params))
        fn = gbl.gpu_logL(devices=devices, use_pmap=True)
        multi = float(fn(params))
        assert np.isclose(single, multi, rtol=1e-5), (
            f"single={single}, multi={multi}"
        )

    def test_nested_closure_arrays_on_correct_device(self, setup):
        """All JAX arrays in nested closures (including pre-allocated freq arrays) are
        on the device assigned to their group after gpu_logL."""
        gbl, _, devices = setup
        fn = gbl.gpu_logL(devices=devices, use_pmap=True)

        kterm_groups, device_list = _extract_closure_groups(fn)

        assert kterm_groups is not None, "Could not find kterm_groups in closure"
        assert device_list is not None, "Could not find device_list in closure"

        for group, expected_device in zip(kterm_groups, device_list):
            for kterm_fn in group:
                # Collect ALL arrays in the entire closure hierarchy, not just
                # the top level — this catches deeply-nested pre-allocated arrays.
                for arr in _all_closure_arrays(kterm_fn):
                    assert arr.device == expected_device, (
                        f"Expected array on {expected_device}, got {arr.device}"
                    )

    def test_put_closure_arrays_on_device_utility(self):
        """put_closure_arrays_on_device correctly transfers nested JAX arrays."""
        devices = _multi_cpu_devices(2)
        target_device = devices[1]

        # Build a chain of closures simulating makegp_fourier's pattern:
        #   outer fn → middle fn → inner fn with a pre-allocated array
        prealloc = jnp.array(np.ones(5))  # lives on devices[0] by default
        assert prealloc.device == devices[0], "Pre-condition: array starts on devices[0]"

        def inner(params):
            return params['x'] * prealloc

        inner.params = ['x']

        def outer(params):
            return inner(params)

        outer.params = ['x']

        transferred = gpu_utils.put_closure_arrays_on_device(outer, target_device)

        # The pre-allocated array in inner's closure should now be on target_device.
        all_arrs = _all_closure_arrays(transferred)
        assert len(all_arrs) > 0, "Should find at least one JAX array in closure"
        for arr in all_arrs:
            assert arr.device == target_device, (
                f"Expected array on {target_device}, got {arr.device}"
            )

    def test_mixed_no_globalgp_and_globalgp_consistency(self):
        """Pre-allocated freq arrays work correctly in both no-globalgp and globalgp paths."""
        devices = _multi_cpu_devices(2)
        n_psr = 4
        psls_freq = [_make_test_psl_with_prealloc_freq(i) for i in range(n_psr)]
        gbl_no_ggp = ds.GlobalLikelihood(psls_freq)

        psls_freq2 = [_make_test_psl_with_prealloc_freq(i) for i in range(n_psr)]
        globalgp = _make_test_globalgp(n_psr)
        gbl_ggp = ds.GlobalLikelihood(psls_freq2, globalgp=globalgp)

        params = {f'log_amp_{i}': jnp.array(0.1 * i) for i in range(n_psr)}
        params['gp_log_amp'] = jnp.array(-1.0)

        # Both paths should give results matching single-device logL.
        single_no_ggp = float(gbl_no_ggp.logL({k: v for k, v in params.items() if k != 'gp_log_amp'}))
        fn_no_ggp = gbl_no_ggp.gpu_logL(devices=devices, use_pmap=True)
        multi_no_ggp = float(fn_no_ggp({k: v for k, v in params.items() if k != 'gp_log_amp'}))
        assert np.isclose(single_no_ggp, multi_no_ggp, rtol=1e-5), (
            f"no-globalgp: single={single_no_ggp}, multi={multi_no_ggp}"
        )

        single_ggp = float(gbl_ggp.logL(params))
        fn_ggp = gbl_ggp.gpu_logL(devices=devices, use_pmap=True)
        multi_ggp = float(fn_ggp(params))
        assert np.isclose(single_ggp, multi_ggp, rtol=1e-5), (
            f"globalgp: single={single_ggp}, multi={multi_ggp}"
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
