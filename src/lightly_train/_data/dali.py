#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Utilities for decoding images with NVIDIA DALI.

The functions in this module are imported lazily so that Lightly Train can be
used without NVIDIA DALI installed. Decoding is performed on the GPU to
accelerate loading of JPEG and PNG files when requested via the
``LIGHTLY_TRAIN_ENABLE_DALI`` environment variable.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict

import numpy as np
import torch

__all__ = [
    "DaliRuntimeError",
    "decode_rgb_image",
    "is_available",
]


class DaliRuntimeError(RuntimeError):
    """Error raised when decoding with DALI fails."""


def is_available() -> bool:
    """Returns True if NVIDIA DALI and a CUDA device are available."""

    try:
        import importlib

        importlib.import_module("nvidia.dali")
    except ModuleNotFoundError:
        return False
    except Exception:
        # Import errors other than ModuleNotFoundError (for example missing CUDA
        # drivers) should be treated as DALI being unavailable.
        return False

    return torch.cuda.is_available()


class _DaliDecoder:
    """Wrapper around a DALI pipeline that decodes RGB images."""

    def __init__(self, *, device_id: int) -> None:
        self._device_id = device_id
        self._lock = threading.Lock()

        from nvidia.dali import fn, types
        from nvidia.dali.pipeline import Pipeline

        # We disable asynchronous execution so the image bytes can be released as
        # soon as the pipeline finishes.
        self._pipeline = Pipeline(
            batch_size=1,
            num_threads=2,
            device_id=device_id,
            exec_async=False,
            exec_pipelined=False,
        )
        with self._pipeline:
            self._images = fn.external_source(name="images", dtype=types.UINT8, device="cpu")
            decoded = fn.decoders.image(
                self._images,
                device="mixed",
                output_type=types.RGB,
            )
            self._pipeline.set_outputs(decoded)
        self._pipeline.build()

    def decode(self, image_path: Path) -> np.ndarray:
        from nvidia.dali.backend import TensorListCPU

        if not image_path.exists():
            raise DaliRuntimeError(f"Image '{image_path}' does not exist.")

        encoded = np.frombuffer(image_path.read_bytes(), dtype=np.uint8)
        if encoded.size == 0:
            raise DaliRuntimeError(f"Image '{image_path}' is empty.")

        try:
            with self._lock:
                self._pipeline.feed_input("images", [encoded])
                (decoded,) = self._pipeline.run()
        except RuntimeError as exc:  # pragma: no cover - depends on DALI internals
            raise DaliRuntimeError(str(exc)) from exc

        # decoded can reside on the GPU (TensorListGPU). Convert to CPU first so
        # we can access it as a numpy array.
        if hasattr(decoded, "as_cpu"):
            decoded = decoded.as_cpu()

        if not isinstance(decoded, TensorListCPU):  # pragma: no cover - sanity check
            raise DaliRuntimeError("Unexpected DALI output type.")

        array = decoded.as_array()
        if array.shape[0] != 1:
            raise DaliRuntimeError(
                "DALI returned an unexpected batch size when decoding an image."
            )

        image = array[0]
        # Ensure the result is contiguous and uses the standard uint8 dtype.
        image = np.ascontiguousarray(image, dtype=np.uint8)
        return image


_decoders: Dict[int, _DaliDecoder] = {}
_decoders_lock = threading.Lock()


def _get_decoder(device_id: int) -> _DaliDecoder:
    with _decoders_lock:
        decoder = _decoders.get(device_id)
        if decoder is None:
            decoder = _DaliDecoder(device_id=device_id)
            _decoders[device_id] = decoder
        return decoder


def decode_rgb_image(image_path: Path) -> np.ndarray:
    """Decodes an RGB image using DALI and returns it as an ``np.ndarray``.

    Args:
        image_path:
            Path to the image that should be decoded.

    Returns:
        The decoded image as ``(H, W, C)`` numpy array with dtype ``uint8``.
    """

    if not is_available():
        raise DaliRuntimeError("NVIDIA DALI is not available.")

    try:
        device_id = torch.cuda.current_device()
    except torch.cuda.CudaError as exc:  # pragma: no cover - defensive
        raise DaliRuntimeError("Could not determine CUDA device for DALI decoding.") from exc

    decoder = _get_decoder(device_id)
    return decoder.decode(image_path)
