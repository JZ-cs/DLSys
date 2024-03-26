"""Logic for backend selection"""
import os


BACKEND = os.environ.get("NEEDLE_BACKEND", "nd")


if BACKEND == "nd":
    from . import backend_ndarray as array_api
    from .backend_ndarray import (
        all_devices,
        cuda,
        cpu,
        cpu_numpy,
        default_device,
        BackendDevice as Device,
    )

    NDArray = array_api.NDArray
elif BACKEND == "np":
    import numpy as array_api
    from .backend_numpy import all_devices, cpu, default_device, Device

    NDArray = array_api.ndarray
elif BACKEND == "tricuda":
    """"""
    import backend_ndarray.ndarray_backend_tricuda as array_api
    from backend_ndarray.ndarray_tricuda import (
        tricuda,
        all_devices, 
        default_device, 
        BackendDevice as Device,
    )
else:
    raise RuntimeError("Unknown needle array backend %s" % BACKEND)
