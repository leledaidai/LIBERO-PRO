# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Solution optimizer classes and functions."""

# Standard Library
import pathlib
from abc import abstractmethod
from typing import Any, Optional

# NVIDIA
from lightwheel.srl.abc.distinctive import Distinctive
from lightwheel.srl.abc.task import Task
from lightwheel.srl.abc.task_solver import TaskSolution


class SolutionOptimizer(Distinctive):
    """Abstract base class for optimizing task solutions."""

    def __init__(self, usd_path: str, task: Task, params: Optional[dict] = None, **kwargs: Any):
        """Initialize a new :class:`~srl.abc.solution_optimizer.SolutionOptimizer` object.

        Args:
            usd_path: Path to USD file that describes the scene.
            task: Task to be solved.
            params: Set of parameters that uniquely define the optimizer.
            kwargs: Additional keyword arguments are passed to the parent class,
                :class:`~srl.abc.distinctive.Distinctive`.
        """
        # Initialize parent class
        super().__init__(params=params, **kwargs)

        # Initialize instance attributes
        self._usd_path = pathlib.Path(usd_path).absolute()
        self._scene_dir_path = self._usd_path.parent
        self._task = task

    @abstractmethod
    def optimize(self, solution: TaskSolution) -> Optional[TaskSolution]:
        """Optimize the task solution.

        Args:
            solution: Solution to be optimized.

        Return:
            A task solution data object or None if an improved solution for that task is not found.
        """
