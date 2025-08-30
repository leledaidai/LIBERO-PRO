# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Helper module for USD types."""

# Third Party
from pxr import Sdf


class ValidUsdTypes:
    """Helper class to make discovering valid USD types easier.

    This class is not meant to be instantiated (but it can be). It is meant be used in an
    interactive python session to discover valid USD types. An example usage in an `ipython` session
    would look like this

    ```
    import nvidia.srl.usd.types_helper as types_helper
    types_helper.ValidUsdTypes.<tab complete>
    ```

    The attributes in `ValidUsdTypes` are the types listed on the left hand side of the table on
    this page: https://developer.nvidia.com/usd/tutorials
    """

    pass


for attr_name in dir(Sdf.ValueTypeNames):
    attr_val = getattr(Sdf.ValueTypeNames, attr_name)
    if isinstance(attr_val, Sdf.ValueTypeName):
        setattr(ValidUsdTypes, attr_name, attr_val)
