# Multi-GPU Parallelization Implementation Summary

This document summarizes the multi-GPU parallelization feature added to Discovery.

## Overview

This implementation adds the ability to distribute pulsar timing array likelihood computations across multiple GPUs on a single machine, enabling faster analyses of large pulsar arrays.

## What's New

### 1. GPU Utilities Module (`discovery.gpu_utils`)

A new module providing GPU device management:

- `get_gpu_devices()` - Detect available GPU devices
- `get_num_gpus()` - Get count of available GPUs
- `setup_gpu_environment()` - Configure GPU usage
- `shard_data()` / `unshard_data()` - Distribute data across devices
- `get_optimal_device_count()` - Calculate optimal device count for workload
- `check_multi_gpu_available()` - Check if multi-GPU is available

### 2. Multi-GPU Likelihood Method

Added `gpu_logL()` method to `GlobalLikelihood` class:

```python
# Use all available GPUs
logl_parallel = gbl.gpu_logL()

# Use specific number of GPUs
logl_parallel = gbl.gpu_logL(devices=2)

# Use specific GPU devices
devices = ds.gpu_utils.get_gpu_devices()
logl_parallel = gbl.gpu_logL(devices=devices[:2])
```

### 3. Comprehensive Documentation

- **`docs/multi_gpu_usage.md`**: Complete usage guide with examples
- **README.md**: Updated with quick-start information
- **Code comments**: All functions fully documented

### 4. Test Suite

- **`tests/test_gpu_utils.py`**: Tests for GPU utilities
- **`tests/test_gpu_likelihood.py`**: Tests for multi-GPU likelihood
- Tests work on systems with or without GPUs
- GPU-specific tests properly marked with `@pytest.mark.gpu`

## Key Features

### ✅ Implemented

- **Automatic GPU detection**: Discovers and configures available GPUs
- **Flexible device selection**: Use all GPUs, specify count, or select specific devices
- **Data parallelism**: Distributes pulsar computations across GPUs
- **Error handling**: Comprehensive validation and helpful error messages
- **Compatibility**: Works with existing code, JIT compilation, and MPI parallelization
- **Graceful degradation**: Works on systems without GPUs
- **Zero breaking changes**: All existing code continues to work unchanged

### 📋 Future Enhancements

- ArrayLikelihood multi-GPU support (planned)
- True `jax.pmap` parallelization for simple case (optimization)
- Automatic load balancing for uneven pulsar counts
- Model parallelism (in addition to data parallelism)
- Integration with JAX's newer sharding API

## Usage Pattern

### Basic Usage

```python
import discovery as ds
import glob

# Load pulsars
psrs = [ds.Pulsar.read_feather(f) for f in glob.glob('data/*.feather')]

# Create likelihood (standard way)
Tspan = ds.getspan(psrs)
gbl = ds.GlobalLikelihood(
    [ds.PulsarLikelihood([psr.residuals,
                          ds.makenoise_measurement(psr, psr.noisedict),
                          ds.makegp_timing(psr)]) for psr in psrs],
    ds.makegp_fourier_global(psrs, ds.powerlaw, ds.hd_orf, 14, T=Tspan, name='gw')
)

# Create multi-GPU version
logl_parallel = gbl.gpu_logL()

# Use with JIT
import jax
logl_jit = jax.jit(logl_parallel)

# Evaluate
params = ds.sample_uniform(logl_parallel.params)
result = logl_jit(params)
```

### With MCMC Sampling

```python
# Create multi-GPU likelihood
logl_parallel = jax.jit(gbl.gpu_logL())

# Use with numpyro sampler
from discovery.samplers import numpyro as dsnp

model = dsnp.makemodel(logl_parallel)
sampler = dsnp.makesampler_nuts(model, num_warmup=1000, num_samples=2000)
sampler.run(jax.random.PRNGKey(0))

# Get results
df = sampler.to_df()
```

## Performance Guidelines

### When to Use Multi-GPU

✅ **Good cases**:
- 20+ pulsars
- Complex noise models (many ECORR/EQUAD terms)
- Many frequency components in GPs
- Running MCMC or nested sampling (many likelihood evaluations)
- Available GPU memory is sufficient

❌ **May not benefit**:
- < 10 pulsars
- Very simple noise models
- Fast likelihood evaluations (overhead dominates)
- Limited GPU memory

### Optimal Setup

1. **Device count**: Use `get_optimal_device_count(num_pulsars)` for even distribution
2. **JIT compilation**: Always use `jax.jit()` on the likelihood
3. **Memory**: Monitor GPU memory with `nvidia-smi`
4. **Initialization**: Call `setup_gpu_environment()` before JAX operations

## Implementation Details

### Architecture

- **Data parallelism**: Each GPU processes a subset of pulsars
- **Device grouping**: Pulsars are grouped and assigned to devices
- **Reduction**: Results from all devices are summed
- **JAX integration**: Uses JAX's device management APIs

### Current Limitations

1. **Sequential group evaluation**: Groups are currently evaluated sequentially (future optimization planned)
2. **No automatic memory management**: User must ensure sufficient GPU memory
3. **No dynamic load balancing**: Distribution is static per call
4. **Requires even-ish distribution**: Performance is best when pulsars divide evenly across GPUs

### Compatibility

- **JAX**: Works with JAX's JIT, vmap, grad, etc.
- **MPI**: Can combine with MPI for multi-node scaling
- **Existing code**: No changes required to existing Discovery code

## Testing

### Run Tests

```bash
# All tests (without GPU tests)
pytest tests/test_gpu_utils.py -k "not gpu"

# With GPU tests (requires GPU hardware)
pytest tests/test_gpu_utils.py

# Specific test classes
pytest tests/test_gpu_utils.py::TestGPUDetection -v
```

### Test Coverage

- ✅ Device detection and counting
- ✅ Data sharding/unsharding
- ✅ Optimal device selection
- ✅ Error handling for edge cases
- ✅ Environment setup
- ⚠️ Integration tests (require actual pulsar data, currently placeholders)

## Troubleshooting

### Common Issues

1. **"No GPU devices available"**
   - Solution: Install CUDA and JAX with GPU support: `pip install jax[cuda12]`

2. **"Number of pulsars is less than number of devices"**
   - Solution: Reduce device count or use standard `logL`

3. **Out of memory errors**
   - Solution: Use fewer GPUs (more memory per GPU) or reduce model complexity

4. **Environment variables not taking effect**
   - Solution: Call `setup_gpu_environment()` before any JAX operations

### Debug Mode

```python
import discovery as ds

# Check GPU availability
print(f"GPUs available: {ds.gpu_utils.get_num_gpus()}")
print(f"Devices: {ds.gpu_utils.get_gpu_devices()}")

# Check multi-GPU status
available, count = ds.gpu_utils.check_multi_gpu_available()
print(f"Multi-GPU available: {available}, Count: {count}")
```

## References

- **Full documentation**: `docs/multi_gpu_usage.md`
- **JAX GPU documentation**: https://jax.readthedocs.io/en/latest/gpu_performance_tips.html
- **Discovery repository**: https://github.com/ryanmshannon/discovery

## Version History

- **v0.5+**: Initial multi-GPU implementation
  - GPU utilities module
  - GlobalLikelihood.gpu_logL() method
  - Comprehensive documentation and tests

---

For questions or issues, please open an issue on the GitHub repository.
