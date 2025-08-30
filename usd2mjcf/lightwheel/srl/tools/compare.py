# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Functions to compare different types of quantities."""

# Third Party
import numpy as np

# NVIDIA
from lightwheel.srl.basics.types import Vector


def vector_eq(vec0: Vector, vec1: Vector) -> bool:
    """Compare two vectors for equality."""
    if isinstance(vec0, np.ndarray) and isinstance(vec1, np.ndarray):
        return vec0.shape == vec1.shape and bool(np.all(vec0 == vec1))
    else:
        return type(vec0) == type(vec1) and vec0 == vec1  # noqa: E721
