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
    
    Args:
        gpu_ids: Optional list of GPU IDs to use (e.g., [0, 1, 2]).
                If None, all available GPUs will be used.
    
    Returns:
        List of configured GPU devices.
    
    Example:
        >>> devices = setup_gpu_environment([0, 1])
        >>> print(f"Using {len(devices)} GPUs")
    """
    if gpu_ids is not None:
        # Set CUDA_VISIBLE_DEVICES to restrict to specific GPUs
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
            f"Cannot evenly shard data with shape {data.shape} "
            f"across {num_devices} devices. First dimension ({data.shape[0]}) "
            f"must be divisible by number of devices ({num_devices})."
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
