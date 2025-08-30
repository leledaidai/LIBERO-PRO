# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""A collection of types."""

# Standard Library
import pathlib

# NOTE (roflaherty): `typing_extensions` is needed to support Python < 3.8.
try:
    # Standard Library
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

# Standard Library
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Third Party
import numpy as np

# NOTE (roflaherty): Using `NotRequired` below would all for more control on which keys are required
# and not required.  However, Available in Python 3.11.0
# (https://docs.python.org/3/library/typing.html#typing.Required).  Instead `total=False` is
# currently set, which allows for all keys to be optional.
# from typing import NotRequired


# Anything that can represent a path
# NOTE: Any additional type added to this list must be convertible to a string using the `str()`
# function.
PathLike = Union[str, pathlib.Path]

# The pose of a rigid body. An element of SE(3). Represented as a 4 x 4 matrix.
Pose = np.ndarray

# A general transform. An element of GA(3) (i.e General Affine group in 3 dimensions). Represented
# as a 4 x 4 matrix.
Affine = np.ndarray

# An element of R^n
Vector = Union[np.ndarray, Sequence[float]]

# The axis-aligned lower and upper bound of a subset of R^n
Interval = Tuple[Vector, Vector]


class Twist(TypedDict):
    """The linear and angular velocity of a rigid body.

    An element of se(3). Represented as two 3 dimensional vectors.
    """

    linear: Vector
    angular: Vector


class JointState(TypedDict, total=False):
    """The joint state of an articulated body.

    The position / angle and the linear / angular velocity of each joint in an articulated body. An
    element of R^{2 N}, where N is the number of joints. This represents the articulated body state
    in reduced coordinates.
    """

    name: List[str]
    position: Vector
    # velocity: NotRequired[Vector]
    velocity: Vector


class FrameState(TypedDict, total=False):
    """The state of a frame in an articulated body.

    An element of (SO(3) x se(3))^M, where M is the number of frames. This represents the
    articulated body state in maximal coordinates.
    """

    name: List[str]
    pose: List[Pose]
    # twist: NotRequired[List[Twist]]
    twist: List[Twist]


class AttachedState(TypedDict):
    """The state of a child entity rigidly attached to a parent body's frame at a relative pose.

    When frame is ``None``, the frame is the body's base frame.

    Example: a Franka Panda robot grasping an entity would set
    `body="/world/robot"`, `frame="panda_hand"`, and `pose` to be a valid grasp.
    """

    body: str
    frame: Optional[str]
    pose: Pose


class BodyAction(TypedDict, total=False):
    """The generalized action taken by a body.

    Supports both high-level actions (`type` and `arguments`) as well as
    low-level delta state actions (`delta_pose`, `delta_joint_state`, `delta_attached_state`).

    Example: a robot picking a cup might set the high-level values to be
    `type="pick"`, `arguments={"obj": "cup"}`.
    """

    type: str
    arguments: Dict[str, Any]
    delta_pose: Pose
    delta_joint_state: JointState
    delta_attached_state: AttachedState


class BodyState(TypedDict, total=False):
    """The state of a prim (either a rigid or articulated)."""

    # TODO (roflaherty): store body_type as an enum instead of a str
    # body_type: BodyType
    body_type: str
    # TODO (roflaherty): store pose_type as an enum instead of a str
    # pose_type: PoseType
    pose_type: str
    pose: Pose
    # twist: NotRequired[Twist]
    twist: Twist
    joint_state: JointState
    # frame_state: NotRequired[FrameState]
    frame_state: FrameState
    # attached_state: NotRequired[AttachedState]
    attached_state: AttachedState
    # TODO (cgarrett): restructure to be more explicit
    body_action: BodyAction
    # TODO (cgarrett): restructure to be more explicit
    metadata: Dict[str, Any]


# The state of the world.
WorldState = Dict[str, BodyState]
