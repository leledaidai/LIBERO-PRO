# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Hash related functions."""

# Standard Library
import hashlib
from typing import Any, Optional


def hash_from_str(input_str: str, hash_len: Optional[int] = None) -> str:
    """Create a hash string from a string.

    Args:
        input_str: Input string.
        hash_len: Number of characters in hash output. Defaults to complete hash string.
    """
    input_utf8 = input_str.encode("utf-8")
    hash_alg = hashlib.sha1()
    hash_alg.update(input_utf8)
    hash_str = hash_alg.hexdigest()
    if hash_len is not None:
        hash_str = hash_str[0:hash_len]
    return hash_str


def hash_from_dict(input_dict: dict, **kwargs: Any) -> str:
    """Create a hash string from a dictionary.

    Args:
        input_dict: Input dictionary.
        kwargs: Additional keyword arguments are passed to
            :func:`~srl.util.hash.hash_from_str`.
    """
    input_str = str(sorted(input_dict.items()))
    return hash_from_str(input_str, **kwargs)
