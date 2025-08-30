# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Task classes and functions."""

# Standard Library
from abc import abstractmethod
from typing import Any

# NVIDIA
from lightwheel.srl.abc.distinctive import Distinctive


class Task(Distinctive):
    """Abstract base class for task objects."""

    def __init__(self, **kwargs: Any):
        """Initialize a new :class:`~srl.abc.task_solver.TaskSolver` object.

        Args:
            kwargs: Additional keyword arguments are passed to the parent class,
                :class:`~srl.abc.srl.SRL`.
        """
        # Initialize parent class
        super().__init__(**kwargs)

    @abstractmethod
    def serialize(self) -> str:
        """Serialize the task to a string."""

    @classmethod
    @abstractmethod
    def deserialize(cls, task_str: str) -> "Task":
        """De-serialize a task from a string."""
