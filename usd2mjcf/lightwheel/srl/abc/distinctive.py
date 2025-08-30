# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Distinctive based class."""

# Standard Library
from typing import Any, Optional

# NVIDIA
from lightwheel.srl.abc.srl import SRL
from lightwheel.srl.tools.hash import hash_from_dict


class Distinctive(SRL):
    """A base class to use a parameter dictionary to uniquely define class instances."""

    HASH_LEN = 7

    def __init__(
        self,
        params: Optional[dict] = None,
        **kwargs: Any,
    ):
        """Initialize a new :class:`~srl.abc.param_class.ParamClass` object.

        Args:
            params: Set of parameters that uniquely define the class. Used to create hash string.
            kwargs: Additional keyword arguments are passed to the parent class,
                :class:`~srl.abc.srl.SRL`.
        """
        # Initialize parent class
        super().__init__(**kwargs)

        # Initialize instance attributes
        if params is None:
            params = dict()
        self._params = params

    def param(self, key: str) -> Any:
        """Get value from the params associated with the given key."""
        return self._params[key]

    def params(self) -> dict:
        """Get params attribute."""
        return self._params

    def hash_str(self) -> str:
        """Generate unique hash string for this object."""
        return hash_from_dict(self._params, hash_len=self.HASH_LEN)

    def __repr__(self) -> str:
        """Return the unambiguous string representation of the object."""
        return f"({self.hash_str()}) {self._params}"

    def __str__(self) -> str:
        """Return the readable string representation of the object."""
        return f"{self.__class__.__name__}: {repr(self)}"
