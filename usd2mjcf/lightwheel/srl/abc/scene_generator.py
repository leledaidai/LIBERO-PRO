# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Scene generator classes and functions."""


# Standard Library
from abc import abstractmethod
from typing import Any, Optional

# Third Party
from pxr import Usd

# NVIDIA
from lightwheel.srl.abc.distinctive import Distinctive


class SceneGenerator(Distinctive):
    """An abstract base class for generating scenes."""

    def __init__(
        self,
        name: str,
        params: Optional[dict] = None,
        **kwargs: Any,
    ):
        """Initialize a new :class:`~srl.abc.scene_generator.SceneGenerator` object.

        Args:
            name: Name given to the scene.
            params: Set of parameters that uniquely define the scene.
            kwargs: Additional keyword arguments are passed to the parent class,
                :class:`~srl.abc.srl.SRL`.
        """
        # Initialize parent class
        super().__init__(params=params, **kwargs)

        # Initialize instance attributes
        self._name = name

    def name(self) -> str:
        """Get name attribute."""
        return self._name

    def __str__(self) -> str:
        """Return the readable string representation of the object."""
        return f"{self.__class__.__name__} ({self._name}): {self.__repr__()}"

    @abstractmethod
    def generate(self) -> Usd.Stage:
        """Generate the USD scene.

        Returns:
            Generated USD stage.
        """
