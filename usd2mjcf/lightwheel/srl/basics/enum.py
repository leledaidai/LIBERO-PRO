# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Collection of general SRL Enum types."""
# Standard Library
import enum
from enum import auto  # noqa: F401


class Enum(enum.Enum):
    """Overload of the enum.Enum class to allow for easy printing of Enums."""

    def __str__(self) -> str:
        """Convert to string."""
        return f"{self.__class__.__name__}.{self.name}"

    def __repr__(self) -> str:
        """Convert to default representation."""
        return str(self)


class BodyType(Enum):
    """Enum to denote the body type of an object."""

    RIGID = auto()
    ARTICULATED = auto()


class PoseType(Enum):
    """Enum to denote the pose type of an object."""

    FIXED = auto()
    FLOATING = auto()
