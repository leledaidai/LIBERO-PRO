# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Helper functions for lists."""

# Standard Library
from typing import Any, List


def to_pretty_str(
    input_list: List[Any], prefix: str = "", separator: str = "\n", suffix: str = ""
) -> str:
    r"""Transform a list into a pretty string.

    Args:
        input_list: The list to be transformed.
        prefix: The string that is prepended to each list element.
        separator: The string separator between each element in the list. Defaults to "\\n".
        suffix: The string that is append to each list element.

    Returns:
        The input list as a pretty string.
    """
    return separator.join(f"{prefix}{elem}{suffix}" for elem in input_list)
