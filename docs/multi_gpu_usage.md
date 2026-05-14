# Multi-GPU Parallelization in Discovery

Discovery now supports distributing computation across multiple GPUs using JAX's parallelization capabilities. This can significantly speed up likelihood evaluations for large pulsar timing arrays.

## Overview

The multi-GPU support in Discovery allows you to:

1. **Automatically detect available GPUs** on your system
2. **Distribute pulsar likelihood computations** across multiple GPUs
3. **Control which GPUs to use** via configuration
4. **Scale analyses** to larger datasets efficiently

## Requirements

To use multi-GPU features, you need:

- JAX with CUDA support: `pip install jax[cuda12]` (or appropriate CUDA version)
- Multiple NVIDIA GPUs with CUDA support
- Sufficient GPU memory for your dataset

## Basic Usage

### Check Available GPUs

```python
import discovery as ds

# Check how many GPUs are available
num_gpus = ds.gpu_utils.get_num_gpus()
print(f"Available GPUs: {num_gpus}")

# Get list of GPU devices
devices = ds.gpu_utils.get_gpu_devices()
print(f"Devices: {devices}")
```

### Setup GPU Environment

```python
# Use all available GPUs
devices = ds.gpu_utils.setup_gpu_environment()

# Or specify which GPUs to use (e.g., GPUs 0 and 2)
devices = ds.gpu_utils.setup_gpu_environment(gpu_ids=[0, 2])
```

You can also control GPU visibility using environment variables before importing JAX:

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'  # Use GPUs 0, 1, and 2

import discovery as ds
```

### Using Multi-GPU Likelihood with GlobalLikelihood

The `GlobalLikelihood` class now has a `gpu_logL()` method that creates a parallelized likelihood function:

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
                          ds.makegp_ecorr(psr, psr.noisedict),
                          ds.makegp_timing(psr, variance=1e-14),
                          ds.makegp_fourier(psr, ds.powerlaw, 30, T=Tspan, name='rednoise')
                          ]) for psr in psrs],
    ds.makegp_fourier_global(psrs, ds.powerlaw, ds.hd_orf, 14, T=Tspan, name='gw')
)

# Create multi-GPU parallelized likelihood
# This will automatically use all available GPUs
logl_parallel = gbl.gpu_logL()

# Or specify number of GPUs to use
logl_parallel = gbl.gpu_logL(devices=2)  # Use 2 GPUs

# Or pass specific device list
devices = ds.gpu_utils.get_gpu_devices()
logl_parallel = gbl.gpu_logL(devices=devices[:2])  # Use first 2 GPUs

# Evaluate likelihood (computation distributed across GPUs)
params = ds.sample_uniform(logl_parallel.params)
result = logl_parallel(params)
```

### Performance Considerations

#### Optimal Number of Pulsars per GPU

For best performance:

- The number of pulsars should be **divisible by the number of GPUs**
- Each GPU should have enough work to overcome parallelization overhead
- Rule of thumb: Use 1 GPU per 5-10 pulsars minimum

```python
# Get optimal device count for your data
num_pulsars = len(psrs)
optimal_devices = ds.gpu_utils.get_optimal_device_count(
    num_pulsars, 
    max_devices=4  # Optional: limit maximum devices
)
print(f"Optimal number of devices: {optimal_devices}")

logl_parallel = gbl.gpu_logL(devices=optimal_devices)
```

#### When to Use Multi-GPU

Multi-GPU parallelization is most beneficial when:

✅ **Use multi-GPU when:**
- You have 20+ pulsars
- Each likelihood evaluation is expensive (complex noise models, many frequency components)
- You're running many likelihood evaluations (MCMC, nested sampling)
- You have sufficient GPU memory

❌ **Single GPU may be better when:**
- You have < 10 pulsars
- Likelihood evaluations are very fast
- Communication overhead dominates computation time
- GPU memory is limited

## Combining with Other Parallelization Methods

### Multi-GPU + JIT Compilation

Always JIT compile your likelihood for best performance:

```python
import jax

logl_parallel = gbl.gpu_logL(devices=2)
logl_jitted = jax.jit(logl_parallel)

# First call compiles
result1 = logl_jitted(params)

# Subsequent calls are fast
result2 = logl_jitted(params2)
```

### Multi-GPU + VMAP for Multiple Parameter Sets

Evaluate likelihood at multiple parameter points in parallel:

```python
import jax
import jax.numpy as jnp

# Create multiple parameter sets
param_sets = [ds.sample_uniform(logl_parallel.params) for _ in range(100)]

# Convert to arrays for vmapping
# (This requires careful structuring of parameter dictionaries)

# Evaluate all parameter sets efficiently
# results = jax.vmap(logl_jitted)(param_arrays)
```

### Multi-GPU vs MPI

Discovery supports both multi-GPU (single node, multiple GPUs) and MPI (multiple nodes):

- **Multi-GPU (`gpu_logL`)**: Single machine with multiple GPUs
  - Easier to set up
  - Lower communication overhead
  - Limited to GPUs on one machine

- **MPI (`plogL`)**: Multiple machines/nodes
  - Scales to larger clusters
  - Higher communication overhead
  - Requires MPI setup

For large-scale runs, you can combine both: use MPI across nodes and multi-GPU within each node.

## Troubleshooting

### "No GPU devices available" Error

If you see this error:

1. Check JAX installation: `python -c "import jax; print(jax.devices())"`
2. Verify CUDA is installed: `nvidia-smi`
3. Install JAX with CUDA: `pip install --upgrade "jax[cuda12]"`
4. Check CUDA_VISIBLE_DEVICES: `echo $CUDA_VISIBLE_DEVICES`

### Out of Memory Errors

If you run out of GPU memory:

1. **Use fewer GPUs** to give each more memory
2. **Reduce batch sizes** if using batched operations
3. **Simplify the model** (fewer frequency components, simpler noise models)
4. **Use gradient checkpointing** with `jax.checkpoint` for memory-intensive operations

### "Number of pulsars not divisible by number of devices" Warning

This warning indicates suboptimal load balancing. Solutions:

1. **Adjust device count**: Use `get_optimal_device_count()` to find a better number
2. **Accept the warning**: Some devices will do slightly more work, but it still works
3. **Pad your dataset**: Add dummy pulsars (not recommended for production)

## Examples

### Example 1: Simple Multi-GPU Likelihood Evaluation

```python
import discovery as ds
import glob

# Load data
psrs = [ds.Pulsar.read_feather(f) for f in glob.glob('data/*.feather')[:20]]
Tspan = ds.getspan(psrs)

# Build likelihood
gbl = ds.GlobalLikelihood(
    [ds.PulsarLikelihood([psr.residuals,
                          ds.makenoise_measurement(psr, psr.noisedict),
                          ds.makegp_timing(psr)]) for psr in psrs]
)

# Setup multi-GPU
num_gpus = ds.gpu_utils.get_num_gpus()
print(f"Using {num_gpus} GPUs")

logl = gbl.gpu_logL()
params = ds.sample_uniform(logl.params)

# Evaluate
import jax
logl_jit = jax.jit(logl)
result = logl_jit(params)
print(f"Log-likelihood: {result}")
```

### Example 2: Multi-GPU with MCMC Sampling

```python
import discovery as ds
import jax

# Setup (same as above)
# ...

# Create multi-GPU likelihood
logl_parallel = gbl.gpu_logL()
logl_jit = jax.jit(logl_parallel)

# Create prior
logp = ds.makelogprior_uniform(logl_parallel.params)
logp_jit = jax.jit(logp)

# Use with numpyro sampler
from discovery.samplers import numpyro as dsnp

model = dsnp.makemodel(logl_jit)
sampler = dsnp.makesampler_nuts(
    model, 
    num_warmup=1000, 
    num_samples=2000,
    num_chains=4
)

# Run sampling (each chain uses multi-GPU for likelihood)
import jax.random
sampler.run(jax.random.PRNGKey(0))

# Get results
df = sampler.to_df()
print(df.describe())
```

### Example 3: Comparing Single-GPU vs Multi-GPU Performance

```python
import discovery as ds
import jax
import time
import glob

# Load data
psrs = [ds.Pulsar.read_feather(f) for f in glob.glob('data/*.feather')]
Tspan = ds.getspan(psrs)

gbl = ds.GlobalLikelihood(
    [ds.PulsarLikelihood([psr.residuals,
                          ds.makenoise_measurement(psr, psr.noisedict),
                          ds.makegp_timing(psr)]) for psr in psrs]
)

params = ds.sample_uniform(gbl.logL.params)

# Single GPU (standard likelihood)
logl_single = jax.jit(gbl.logL)
_ = logl_single(params)  # Warmup

start = time.time()
for _ in range(100):
    _ = logl_single(params)
single_time = time.time() - start

# Multi-GPU
num_gpus = ds.gpu_utils.get_num_gpus()
logl_multi = jax.jit(gbl.gpu_logL())
_ = logl_multi(params)  # Warmup

start = time.time()
for _ in range(100):
    _ = logl_multi(params)
multi_time = time.time() - start

print(f"Single GPU: {single_time:.3f}s for 100 evaluations")
print(f"Multi-GPU ({num_gpus} GPUs): {multi_time:.3f}s for 100 evaluations")
print(f"Speedup: {single_time/multi_time:.2f}x")
```

## Advanced Topics

### Custom Device Placement

For fine-grained control, you can manually place data on specific devices:

```python
import jax

# Get devices
devices = ds.gpu_utils.get_gpu_devices()

# Place data on specific device
data_gpu0 = jax.device_put(data, devices[0])
data_gpu1 = jax.device_put(data, devices[1])
```

### Memory Management

Monitor GPU memory usage:

```python
# Before running
!nvidia-smi

# In Python, check device memory
for i, device in enumerate(ds.gpu_utils.get_gpu_devices()):
    print(f"GPU {i}: {device}")
```

### Using with JAX Sharding (Advanced)

For very large models, consider using JAX's newer sharding API:

```python
from jax.sharding import PositionalSharding

# Create sharding for data distribution
sharding = PositionalSharding(ds.gpu_utils.get_gpu_devices())

# Shard arrays across devices
# (Requires more advanced JAX knowledge)
```

## Best Practices

1. **Profile first**: Always profile single-GPU performance before parallelizing
2. **JIT everything**: Use `jax.jit` on your likelihood and prior functions
3. **Minimize host-device transfers**: Keep data on GPU as much as possible
4. **Balance workload**: Ensure even distribution of work across GPUs
5. **Monitor memory**: Use `nvidia-smi` to track GPU memory usage
6. **Start small**: Test with a subset of data before scaling up

## API Reference

See `discovery.gpu_utils` module for detailed API documentation:

- `get_gpu_devices()`: Get list of available GPU devices
- `get_num_gpus()`: Get number of available GPUs
- `setup_gpu_environment(gpu_ids)`: Configure GPU environment
- `check_multi_gpu_available()`: Check if multi-GPU is available
- `get_optimal_device_count(num_tasks, max_devices)`: Get optimal device count
- `shard_data(data, num_devices)`: Shard data across devices
- `unshard_data(sharded_data)`: Unshard data back to single array

## Future Improvements

Planned enhancements for multi-GPU support:

- [ ] Automatic load balancing for uneven pulsar counts
- [ ] Support for model parallelism (in addition to data parallelism)
- [ ] Integration with distributed training frameworks
- [ ] Better memory management for large datasets
- [ ] Automatic device selection based on available memory
- [ ] Support for mixed CPU/GPU execution

## References

- JAX documentation on parallelism: https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html
- JAX distributed arrays: https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html
- Discovery GitHub repository: https://github.com/ryanmshannon/discovery
