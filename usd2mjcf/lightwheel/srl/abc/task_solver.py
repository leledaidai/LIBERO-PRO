# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Task solver classes and functions."""

# Standard Library
import pathlib
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

# NVIDIA
from lightwheel.srl.abc.distinctive import Distinctive
from lightwheel.srl.abc.task import Task
from lightwheel.srl.basics.types import WorldState


# TODO (roflaherty): Convert this to a child class of `srl.abc.trajectory.Trajectory`
@dataclass
class TaskSolution:
    """Data class that holds a solution for a task."""

    times: List[float]
    states: List[WorldState]

    def start_time(self) -> float:
        """Get the start time of the solution."""
        return self.times[0]

    def end_time(self) -> float:
        """Get the end time of the solution."""
        return self.times[-1]

    def start_state(self) -> WorldState:
        """Get the start state of the solution."""
        return self.states[0]

    def end_state(self) -> WorldState:
        """Get the end state of the solution."""
        return self.states[-1]

    def duration(self) -> float:
        """Get the time duration the solution."""
        return self.end_time() - self.start_time()

    def length(self) -> int:
        """Get the number of states in the solution."""
        return len(self.states)

    def __len__(self) -> int:
        """Get the number of states in the solution."""
        return self.length()

    def __iter__(self) -> "TaskSolution":
        """Return itself as an iterator."""
        self._current_index = 0
        return self

    def __next__(self) -> Tuple[float, WorldState]:
        """Return the next element in the iteration of itself."""
        if self._current_index < self.length():
            time = self.times[self._current_index]
            state = self.states[self._current_index]
            self._current_index += 1
            return (time, state)
        raise StopIteration


class SolverError(Exception):
    """An exception that occurs when solving a task."""

    pass


class TaskSolver(Distinctive):
    """Abstract base class for solving tasks."""

    def __init__(self, usd_path: str, task: Task, params: Optional[dict] = None, **kwargs: Any):
        """Initialize a new :class:`~srl.abc.task_solver.TaskSolver` object.

        Args:
            usd_path: Path to USD file that describes the scene.
            task: Task to be solved.
            params: Set of parameters that uniquely define the solver.
            kwargs: Additional keyword arguments are passed to the parent class,
                :class:`~srl.abc.srl.SRL`.
        """
        # Initialize parent class
        super().__init__(params=params, **kwargs)

        # Initialize instance attributes
        self._usd_path = pathlib.Path(usd_path).absolute()
        self._scene_dir_path = self._usd_path.parent
        self._task = task

    @abstractmethod
    def solve(self) -> Optional[TaskSolution]:
        """Solve the task.

        Return:
            A task solution data object or None if a solution for that task is not found.
        """
