# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""General test helper functions."""

# Standard Library
from inspect import getframeinfo, stack
from typing import Optional

# Third Party
import numpy as np

# NVIDIA
from lightwheel.srl.basics.types import Vector


def assert_np_allclose(
    expected: Vector,
    actual: Vector,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    equal_nan: bool = False,
    additional_msg: Optional[str] = None,
) -> None:
    """Wrap the `assert np.allclose` to make it automatically print in a readable form.

    Args:
        expected: Expected array to compare against the actual array.
        actual: Actual array.
        rtol: The relative tolerance parameter (see notes in :func:`numpy.allclose`).
        atol: The absolute tolerance parameter (see Notes in :func:`numpy.allclose`).
        equal_nan: Whether to compare NaN's as equal.  If True, NaN's in `expected` will be
            considered equal to NaN's in `actual` in the output array.
        additional_msg: An additional message to appended to the default message that is printed if
            the assert fails.
    """
    caller = getframeinfo(stack()[1][0])
    with np.printoptions(suppress=True):
        failed_str = "\n".join(
            [
                f"{caller.filename}, {caller.lineno}",
                f"expected:\n{expected}\nactual:\n{actual}",
                f"rtol: {rtol}, atol: {atol}, equal_nan: {equal_nan}",
            ]
        )
        if additional_msg is not None:
            failed_str = "\n".join([failed_str, additional_msg])
    assert np.allclose(expected, actual, rtol, atol, equal_nan), failed_str
