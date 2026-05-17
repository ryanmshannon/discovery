"""GPU utilities for multi-device parallelization in Discovery.

This module provides utilities for detecting and managing multiple GPUs,
and for distributing computation across them using JAX's parallelization
primitives (pmap, device_put, etc.).
"""

import os
import warnings
from typing import List, Optional, Tuple, Any

import jax
import jax.numpy as jnp


def get_gpu_devices() -> List[Any]:
    """Get list of available GPU devices.
    
    Returns:
        List of JAX GPU devices, or empty list if no GPUs available.
    """
    try:
        devices = jax.devices('gpu')
        return devices
    except RuntimeError:
        # No GPUs available
        return []


def get_num_gpus() -> int:
    """Get the number of available GPU devices.
    
    Returns:
        Number of GPU devices available.
    """
    return len(get_gpu_devices())


def setup_gpu_environment(gpu_ids: Optional[List[int]] = None) -> List[Any]:
    """Setup GPU environment for multi-GPU execution.
    
    This function configures JAX to use specific GPUs if requested,
    or all available GPUs otherwise.
    
    **Important**: Call this function before importing or using JAX for the first time,
    or before any JAX operations that would initialize CUDA. Setting environment
    variables after JAX has initialized devices may not take effect.
    
    Args:
        gpu_ids: Optional list of GPU IDs to use (e.g., [0, 1, 2]).
                If None, all available GPUs will be used.
    
    Returns:
        List of configured GPU devices.
    
    Example:
        >>> # Call this early in your script, before JAX operations
        >>> devices = setup_gpu_environment([0, 1])
        >>> print(f"Using {len(devices)} GPUs")
    """
    if gpu_ids is not None:
        # Set CUDA_VISIBLE_DEVICES to restrict to specific GPUs
        # Warn if JAX may have already initialized
        try:
            # Check if JAX has already created devices
            existing_devices = jax.devices()
            if len(existing_devices) > 0:
                warnings.warn(
                    "JAX has already initialized devices. Setting CUDA_VISIBLE_DEVICES "
                    "may not take effect. For best results, call setup_gpu_environment() "
                    "before any JAX operations.",
                    UserWarning
                )
        except Exception:
            # If we can't check, proceed anyway
            pass
        
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
    
    devices = get_gpu_devices()
    
    if len(devices) == 0:
        warnings.warn("No GPU devices found. Computations will run on CPU.")
    else:
        print(f"Discovery: Using {len(devices)} GPU device(s): {devices}")
    
    return devices


def shard_data(data: jnp.ndarray, num_devices: int) -> jnp.ndarray:
    """Shard data for distribution across multiple devices.
    
    Args:
        data: Input array to shard. First dimension should be divisible
              by num_devices for even distribution.
        num_devices: Number of devices to shard across.
    
    Returns:
        Reshaped array with shape (num_devices, batch_per_device, ...).
        
    Raises:
        ValueError: If first dimension is not divisible by num_devices.
    
    Example:
        >>> data = jnp.arange(8).reshape(8, 1)
        >>> sharded = shard_data(data, 2)  # Shape: (2, 4, 1)
    """
    if data.shape[0] % num_devices != 0:
        raise ValueError(
            f"Data with shape {data.shape} is not evenly divisible "
            f"across {num_devices} devices (first dimension {data.shape[0]} "
            f"must be divisible by {num_devices})."
        )
    
    batch_per_device = data.shape[0] // num_devices
    new_shape = (num_devices, batch_per_device) + data.shape[1:]
    return data.reshape(new_shape)


def unshard_data(sharded_data: jnp.ndarray) -> jnp.ndarray:
    """Unshard data from multi-device format back to single array.
    
    Args:
        sharded_data: Sharded array with shape (num_devices, batch_per_device, ...).
    
    Returns:
        Unsharded array with shape (num_devices * batch_per_device, ...).
        
    Example:
        >>> sharded = jnp.arange(8).reshape(2, 4, 1)
        >>> unsharded = unshard_data(sharded)  # Shape: (8, 1)
    """
    num_devices = sharded_data.shape[0]
    batch_per_device = sharded_data.shape[1]
    new_shape = (num_devices * batch_per_device,) + sharded_data.shape[2:]
    return sharded_data.reshape(new_shape)


def distribute_to_devices(data: Any, devices: Optional[List[Any]] = None) -> List[Any]:
    """Distribute data to specific devices using device_put.
    
    Args:
        data: Data to distribute (can be arrays, nested structures, etc.).
        devices: List of devices to distribute to. If None, uses all GPUs.
    
    Returns:
        List of data placed on each device.
        
    Example:
        >>> devices = get_gpu_devices()
        >>> data = jnp.array([1, 2, 3, 4])
        >>> distributed = distribute_to_devices(data, devices[:2])
    """
    if devices is None:
        devices = get_gpu_devices()
        if len(devices) == 0:
            devices = jax.devices('cpu')
    
    return [jax.device_put(data, device) for device in devices]


def check_multi_gpu_available() -> Tuple[bool, int]:
    """Check if multi-GPU execution is available.
    
    Returns:
        Tuple of (is_available, num_gpus).
    """
    num_gpus = get_num_gpus()
    return (num_gpus > 1, num_gpus)


def get_optimal_device_count(num_tasks: int, max_devices: Optional[int] = None) -> int:
    """Get optimal number of devices to use for given number of tasks.
    
    Args:
        num_tasks: Number of independent tasks to parallelize.
        max_devices: Maximum number of devices to use. If None, uses all available.
    
    Returns:
        Optimal number of devices (will evenly divide num_tasks if possible).
    """
    available_devices = get_num_gpus()
    if available_devices == 0:
        return 1
    
    if max_devices is not None:
        available_devices = min(available_devices, max_devices)
    
    # Find largest divisor of num_tasks that's <= available_devices
    for n_devices in range(min(num_tasks, available_devices), 0, -1):
        if num_tasks % n_devices == 0:
            return n_devices
    
    # If no even divisor, return available_devices
    return available_devices


def pmap_wrapper(func, in_axes=0, out_axes=0, devices=None):
    """Wrapper for jax.pmap that handles device selection.
    
    Args:
        func: Function to parallelize.
        in_axes: Input axes to map over (see jax.pmap docs).
        out_axes: Output axes to map over (see jax.pmap docs).
        devices: Optional list of devices to use.
    
    Returns:
        Parallelized function using jax.pmap.
    """
    if devices is None:
        devices = get_gpu_devices()
        if len(devices) == 0:
            warnings.warn("No GPUs available, using CPU for pmap.")
            devices = None  # Let JAX choose
    
    return jax.pmap(func, in_axes=in_axes, out_axes=out_axes, devices=devices)


def create_sharding_spec(num_devices: int, data_axis: int = 0) -> Any:
    """Create a JAX sharding specification for distributing data across devices.
    
    This function creates a PositionalSharding that specifies how to distribute
    data across multiple devices. The sharding is applied along a specific axis
    of the data.
    
    Args:
        num_devices: Number of devices to shard across.
        data_axis: Axis along which to shard the data (default: 0).
    
    Returns:
        A JAX PositionalSharding object.
        
    Example:
        >>> sharding = create_sharding_spec(4, data_axis=0)
        >>> # Use with jax.device_put to place sharded data on devices
    """
    try:
        from jax.sharding import PositionalSharding
        devices = get_gpu_devices()
        if len(devices) == 0:
            devices = jax.devices('cpu')
        
        # Ensure we have enough devices
        if len(devices) < num_devices:
            warnings.warn(
                f"Requested {num_devices} devices but only {len(devices)} available. "
                f"Using {len(devices)} devices instead."
            )
            num_devices = len(devices)
        
        devices = devices[:num_devices]
        sharding = PositionalSharding(devices)
        
        return sharding
    except ImportError:
        # Fall back to None if PositionalSharding is not available (older JAX)
        warnings.warn(
            "JAX PositionalSharding not available. Sharding functionality requires JAX >= 0.4.0"
        )
        return None


def shard_array_to_devices(array: jnp.ndarray, num_devices: int, axis: int = 0) -> jnp.ndarray:
    """Shard an array across multiple devices using JAX's device placement.
    
    This function distributes an array across multiple GPU devices, allowing
    for true parallel computation. Unlike `shard_data` which just reshapes,
    this function actually places data on different physical devices.
    
    Args:
        array: Array to shard across devices.
        num_devices: Number of devices to shard across.
        axis: Axis along which to shard (default: 0).
    
    Returns:
        Array with data distributed across devices.
        
    Example:
        >>> data = jnp.arange(100).reshape(10, 10)
        >>> sharded = shard_array_to_devices(data, 2)  # Distribute across 2 GPUs
    """
    try:
        from jax.sharding import PositionalSharding
        
        if array.shape[axis] % num_devices != 0:
            raise ValueError(
                f"Array dimension {array.shape[axis]} along axis {axis} "
                f"is not evenly divisible by {num_devices} devices."
            )
        
        devices = get_gpu_devices()
        if len(devices) == 0:
            warnings.warn("No GPUs available. Array will remain on CPU.")
            return array
        
        devices = devices[:min(num_devices, len(devices))]
        sharding = PositionalSharding(devices)
        
        # For simple 1D device sharding, we can use direct placement
        # More complex sharding patterns would require mesh-based sharding
        # For now, use simple device_put approach
        return jax.device_put(array, sharding)
        
    except (ImportError, AttributeError):
        # Fallback for older JAX versions
        warnings.warn("Advanced sharding not available. Using simple device placement.")
        return array


def replicate_across_devices(data: Any, devices: Optional[List[Any]] = None) -> Any:
    """Replicate data across all specified devices.
    
    This is useful for parameters or constants that need to be available
    on all devices for parallel computation.
    
    Args:
        data: Data to replicate (can be arrays, nested structures, etc.).
        devices: List of devices to replicate to. If None, uses all GPUs.
    
    Returns:
        Data replicated across devices.
        
    Example:
        >>> params = {'log10_A': -15.0, 'gamma': 4.33}
        >>> replicated = replicate_across_devices(params)
    """
    if devices is None:
        devices = get_gpu_devices()
        if len(devices) == 0:
            devices = jax.devices('cpu')
    
    # For pmap, we need to stack the data along the first axis
    # and then JAX will automatically place each slice on a device
    return jax.tree_util.tree_map(
        lambda x: jnp.stack([x] * len(devices)) if isinstance(x, jnp.ndarray) else x,
        data
    )


def pmap_reduce_sum(func, devices=None):
    """Create a pmapped function that sums results across devices.
    
    This is a common pattern for likelihood computations where each device
    computes a partial likelihood and results need to be summed.
    
    Args:
        func: Function to parallelize. Should take parameters and return a scalar.
        devices: Optional list of devices to use.
    
    Returns:
        A function that distributes computation and sums results.
        
    Example:
        >>> def compute_partial(params, data):
        ...     return jnp.sum(data * params['scale'])
        >>> parallel_func = pmap_reduce_sum(compute_partial)
    """
    if devices is None:
        devices = get_gpu_devices()
        if len(devices) == 0:
            warnings.warn("No GPUs available, using CPU for pmap.")
            devices = jax.devices('cpu')
    
    # Create pmapped version
    pmapped = jax.pmap(func, devices=devices)
    
    # Wrapper that sums results
    def wrapper(*args, **kwargs):
        results = pmapped(*args, **kwargs)
        return jnp.sum(results)
    
    return wrapper


def get_device_mesh(num_devices: Optional[int] = None) -> Any:
    """Create a device mesh for advanced sharding patterns.
    
    A device mesh is useful for multi-dimensional sharding patterns,
    e.g., sharding both data and model parameters.
    
    Args:
        num_devices: Number of devices to include in mesh. If None, uses all GPUs.
    
    Returns:
        A JAX Mesh object (or None if not available).
        
    Example:
        >>> mesh = get_device_mesh(4)
        >>> # Use with NamedSharding for complex sharding patterns
    """
    try:
        from jax.sharding import Mesh
        from jax.experimental import mesh_utils
        
        devices = get_gpu_devices()
        if len(devices) == 0:
            devices = jax.devices('cpu')
        
        if num_devices is not None:
            devices = devices[:min(num_devices, len(devices))]
        
        # Create a 1D mesh (can be extended to 2D for more complex patterns)
        device_array = mesh_utils.create_device_mesh((len(devices),), devices)
        mesh = Mesh(device_array, ('data',))
        
        return mesh
        
    except ImportError:
        warnings.warn("JAX Mesh not available. Requires JAX >= 0.4.0")
        return None
