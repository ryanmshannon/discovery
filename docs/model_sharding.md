# Model Sharding for Multi-GPU Distribution

This document describes the model sharding capabilities in Discovery for distributing complex models across multiple GPUs.

## Overview

Model sharding allows you to split large computational tasks across multiple GPUs, enabling:

1. **Parallel computation** of independent pulsar likelihoods
2. **Device placement** for optimal GPU utilization  
3. **Scalability** to larger datasets and more complex models
4. **Memory efficiency** by distributing data across devices

## What is Sharding?

Sharding is a parallelization strategy where:
- **Data parallelism**: Different pulsars are processed on different GPUs
- **Model parallelism**: Large model components can be split across devices
- **Device placement**: Computations are explicitly placed on specific GPUs

Discovery implements **data parallelism** for pulsar timing array likelihoods, where each GPU processes a subset of pulsars.

## Quick Start

### Basic Usage

```python
import discovery as ds
import glob

# Load pulsars
psrs = [ds.Pulsar.read_feather(f) for f in glob.glob('data/*.feather')]

# Create likelihood
Tspan = ds.getspan(psrs)
gbl = ds.GlobalLikelihood(
    [ds.PulsarLikelihood([psr.residuals,
                          ds.makenoise_measurement(psr, psr.noisedict),
                          ds.makegp_timing(psr)]) for psr in psrs]
)

# Create sharded likelihood (automatic multi-GPU)
logl_sharded = gbl.gpu_logL(use_pmap=True)

# JIT compile for best performance
import jax
logl_jit = jax.jit(logl_sharded)

# Evaluate
params = ds.sample_uniform(logl_sharded.params)
result = logl_jit(params)
```

### With Global GP (HD Model)

```python
# HD likelihood with sharding
gbl = ds.GlobalLikelihood(
    [ds.PulsarLikelihood([psr.residuals,
                          ds.makenoise_measurement(psr, psr.noisedict),
                          ds.makegp_fourier(psr, ds.powerlaw, 30, T=Tspan, name='rednoise')
                          ]) for psr in psrs],
    ds.makegp_fourier_global(psrs, ds.powerlaw, ds.hd_orf, 14, T=Tspan, name='gw')
)

# Create sharded version
logl_sharded = gbl.gpu_logL(devices=4, use_pmap=True)
```

## How It Works

### GlobalLikelihood Sharding

For `GlobalLikelihood` without `globalgp`:
1. Pulsars are divided into groups (one per GPU)
2. Each group's likelihoods are computed on its assigned GPU
3. Results are summed across all devices

For `GlobalLikelihood` with `globalgp`:
1. Per-pulsar kernel terms are computed in parallel on different GPUs
2. Global GP operations (matrix inversions, etc.) are done on the primary GPU
3. Device placement ensures data stays on appropriate GPUs during computation

### ArrayLikelihood Sharding

`ArrayLikelihood` uses batched operations that are already GPU-optimized. The `gpu_logL()` method primarily handles device placement:

```python
arl = ds.ArrayLikelihood(psls, commongp=commongp)
logl_gpu = arl.gpu_logL(devices=1)  # Best on single large GPU
```

## Configuration Options

### Specifying Devices

```python
# Use all available GPUs
logl = gbl.gpu_logL()

# Use specific number of GPUs
logl = gbl.gpu_logL(devices=2)

# Use specific GPU devices
devices = ds.gpu_utils.get_gpu_devices()
logl = gbl.gpu_logL(devices=[devices[0], devices[2]])  # Use GPUs 0 and 2
```

### Sharding Mode

```python
# With device placement (default, recommended)
logl = gbl.gpu_logL(use_pmap=True)

# Without device placement (fallback, for debugging)
logl = gbl.gpu_logL(use_pmap=False)
```

### Environment Control

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'  # Use GPUs 0, 1, 2

import discovery as ds
# Or use setup_gpu_environment
ds.gpu_utils.setup_gpu_environment([0, 1, 2])
```

## Performance Optimization

### When to Use Sharding

✅ **Sharding is beneficial when:**
- You have 20+ pulsars
- Each likelihood evaluation is computationally expensive
- You have multiple GPUs with sufficient memory
- Running many likelihood evaluations (MCMC, nested sampling)

❌ **Single GPU may be better when:**
- You have < 10 pulsars
- Likelihood evaluations are very fast
- Limited GPU memory
- GPU-to-GPU communication overhead dominates

### Optimal Device Count

```python
# Get recommended device count
num_pulsars = len(psrs)
optimal_devices = ds.gpu_utils.get_optimal_device_count(
    num_pulsars,
    max_devices=4
)
print(f"Recommended: {optimal_devices} devices")

logl = gbl.gpu_logL(devices=optimal_devices)
```

### Best Practices

1. **Even distribution**: Best performance when pulsars divide evenly across GPUs
   ```python
   # Good: 24 pulsars, 4 GPUs = 6 per GPU
   # Suboptimal: 22 pulsars, 4 GPUs = 5.5 per GPU (uneven)
   ```

2. **JIT compilation**: Always JIT your sharded likelihood
   ```python
   logl_jit = jax.jit(logl_sharded)
   _ = logl_jit(params)  # First call compiles
   ```

3. **Memory management**: Monitor GPU memory usage
   ```bash
   watch -n 1 nvidia-smi
   ```

4. **Batch size**: For MCMC, run multiple chains on different GPUs
   ```python
   # Each chain can use multi-GPU likelihood internally
   # Or distribute chains across GPUs
   ```

## Advanced Features

### Custom Sharding Utilities

Discovery provides low-level sharding utilities for advanced users:

```python
from discovery import gpu_utils

# Create sharding specification
sharding = gpu_utils.create_sharding_spec(num_devices=4, data_axis=0)

# Shard array across devices
data = jnp.arange(1000).reshape(100, 10)
sharded_data = gpu_utils.shard_array_to_devices(data, num_devices=4)

# Replicate data across devices
replicated = gpu_utils.replicate_across_devices(params)

# Create device mesh for complex sharding
mesh = gpu_utils.get_device_mesh(num_devices=4)
```

### JAX Transformations

Sharded likelihoods work with JAX transformations:

```python
# JIT compilation
logl_jit = jax.jit(logl_sharded)

# Gradient computation (for HMC)
grad_logl = jax.jit(jax.grad(logl_sharded))

# Vmap over parameter sets
param_sets = [ds.sample_uniform(logl.params) for _ in range(10)]
# vmapping requires careful parameter structure handling
```

## Combining with Other Parallelization

### Multi-GPU + MPI

For cluster computing, combine GPU sharding with MPI:

```python
# Within each MPI process, use multi-GPU
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Each MPI process can use multiple GPUs
logl_parallel = gbl.gpu_logL(devices=2)  # 2 GPUs per MPI process
```

### Multi-GPU + JAX vmap

For batch evaluation:

```python
# Evaluate at multiple parameter points
# (requires careful parameter structuring)
logl_jit = jax.jit(logl_sharded)
# Use vmap if parameters can be vectorized
```

## Troubleshooting

### "No GPU devices available"

**Solution**: Verify CUDA installation and JAX GPU support
```bash
python -c "import jax; print(jax.devices())"
nvidia-smi
pip install --upgrade "jax[cuda12]"
```

### "Number of pulsars less than number of devices"

**Solution**: Reduce device count or use standard `logL`
```python
num_pulsars = len(psrs)
safe_devices = min(num_pulsars, ds.gpu_utils.get_num_gpus())
logl = gbl.gpu_logL(devices=safe_devices)
```

### Out of GPU Memory

**Solutions**:
1. Use fewer GPUs (more memory per GPU)
2. Reduce model complexity (fewer frequency components)
3. Use gradient checkpointing
   ```python
   from jax import checkpoint
   logl_checkpointed = checkpoint(logl_sharded)
   ```
4. Decrease batch size in MCMC samplers

### Numerical Differences

Sharded and non-sharded likelihoods should match within floating-point precision:

```python
# Verify results match
logl_standard = gbl.logL
logl_sharded = gbl.gpu_logL()

result_std = logl_standard(params)
result_sharded = logl_sharded(params)

assert jnp.allclose(result_std, result_sharded, rtol=1e-6)
```

### Performance Not Improving

Check:
1. **Problem size**: Too few pulsars for multi-GPU overhead
2. **GPU utilization**: Run `nvidia-smi dmon` during computation
3. **Memory bandwidth**: Bottleneck may be data transfer
4. **Computation bottleneck**: Profile with JAX profiler

## Performance Benchmarking

### Basic Benchmark

```python
import time
import jax

# Standard version
logl_std = jax.jit(gbl.logL)
_ = logl_std(params)  # Warmup

start = time.time()
for _ in range(100):
    _ = logl_std(params).block_until_ready()
time_std = time.time() - start

# Sharded version
logl_sharded = jax.jit(gbl.gpu_logL(devices=4))
_ = logl_sharded(params)  # Warmup

start = time.time()
for _ in range(100):
    _ = logl_sharded(params).block_until_ready()
time_sharded = time.time() - start

print(f"Standard: {time_std:.3f}s")
print(f"Sharded (4 GPUs): {time_sharded:.3f}s")
print(f"Speedup: {time_std/time_sharded:.2f}x")
```

### Detailed Profiling

```python
# Use JAX profiler for detailed analysis
with jax.profiler.trace("/tmp/jax-trace"):
    _ = logl_sharded(params)

# View trace in TensorBoard
# tensorboard --logdir=/tmp/jax-trace
```

## Examples

See `examples/` directory for complete examples:
- `sharding_example.py`: Basic sharding usage
- `sharding_hd_example.py`: HD model with sharding
- `sharding_benchmark.py`: Performance benchmarking

## API Reference

### GlobalLikelihood.gpu_logL()

```python
def gpu_logL(self, devices=None, use_pmap=True):
    """Create multi-GPU parallelized likelihood.
    
    Args:
        devices: None (all GPUs), int (number), or list (specific devices)
        use_pmap: bool, whether to use device placement (default: True)
    
    Returns:
        Parallelized likelihood function
    """
```

### ArrayLikelihood.gpu_logL()

```python
def gpu_logL(self, devices=None, use_pmap=True):
    """Create GPU-placed likelihood for ArrayLikelihood.
    
    Args:
        devices: None (all GPUs), int (number), or list (specific devices)
        use_pmap: bool, currently for compatibility (default: True)
    
    Returns:
        GPU-placed likelihood function
    """
```

### GPU Utilities

See `discovery.gpu_utils` module documentation for full API.

## Technical Details

### Memory Layout

- **Data parallelism**: Each GPU stores a subset of pulsar data
- **Parameter replication**: Model parameters replicated to all GPUs
- **Result aggregation**: Partial results combined on host

### Communication Patterns

1. **Broadcast**: Parameters sent to all GPUs
2. **Scatter**: Pulsar data distributed to GPUs
3. **Reduce**: Likelihood results summed

### JAX Backend

Discovery uses JAX's device placement (`jax.device_put`) and context managers (`jax.default_device`) for sharding, which provides:
- Explicit control over device placement
- Compatibility with JIT compilation
- Support for heterogeneous device configurations

## Future Enhancements

Planned improvements:
- [ ] True `jax.pmap` parallelization for simple case
- [ ] Model parallelism for very large GPs
- [ ] Automatic load balancing for uneven pulsar counts
- [ ] Integration with JAX's new sharding API
- [ ] Support for distributed arrays
- [ ] Automatic device selection based on available memory

## References

- [JAX Device Parallelism](https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html)
- [JAX Sharding](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)
- Discovery `multi_gpu_usage.md` for basic multi-GPU usage

## Support

For questions or issues:
1. Check this documentation
2. Review `multi_gpu_usage.md` for basic multi-GPU setup
3. Open an issue on GitHub with:
   - Your GPU configuration (`nvidia-smi`)
   - JAX version and backend
   - Minimal reproducing example
