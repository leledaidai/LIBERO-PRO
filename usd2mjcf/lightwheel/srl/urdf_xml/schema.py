# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Python dataclasses to represent the URDF XML schema.

* URDF XML Schema definition:
    https://raw.githubusercontent.com/ros/urdfdom/master/xsd/urdf.xsd
* ROS.org URDF package:
    http://wiki.ros.org/urdf
"""


# Standard Library
from dataclasses import dataclass, field
from typing import Any, List, Optional

# NVIDIA
from lightwheel.srl.basics.types import Vector
from lightwheel.srl.tools.compare import vector_eq
from lightwheel.srl.urdf_xml.element import UrdfXmlElement


@dataclass
class Pose(UrdfXmlElement):
    """Define the pose XML element."""

    xyz: Vector = field(default_factory=lambda: [0, 0, 0], metadata={"type": "attribute"})
    rpy: Vector = field(default_factory=lambda: [0, 0, 0], metadata={"type": "attribute"})

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Pose)
            and vector_eq(self.xyz, other.xyz)
            and vector_eq(self.rpy, other.rpy)
        )


@dataclass
class Color(UrdfXmlElement):
    """Define the color XML element."""

    rgba: Vector = field(default_factory=lambda: [0, 0, 0, 0], metadata={"type": "attribute"})

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Color) and vector_eq(self.rgba, other.rgba)


@dataclass
class Verbose(UrdfXmlElement):
    """Define the verbose XML element."""

    value: str = field(metadata={"type": "attribute"})


@dataclass
class Name(UrdfXmlElement):
    """Define the name XML element."""

    name: str = field(metadata={"type": "attribute"})


@dataclass
class Mass(UrdfXmlElement):
    """Define the mass XML element."""

    value: float = field(default=0, metadata={"type": "attribute"})


@dataclass
class Inertia(UrdfXmlElement):
    """Define the inertia XML element."""

    ixx: float = field(default=0, metadata={"type": "attribute"})
    ixy: float = field(default=0, metadata={"type": "attribute"})
    ixz: float = field(default=0, metadata={"type": "attribute"})
    iyy: float = field(default=0, metadata={"type": "attribute"})
    iyz: float = field(default=0, metadata={"type": "attribute"})
    izz: float = field(default=0, metadata={"type": "attribute"})


@dataclass
class Inertial(UrdfXmlElement):
    """Define the inertial XML element."""

    origin: Pose = field(default_factory=Pose(), metadata={"type": "element"})
    mass: Mass = field(default_factory=Mass(), metadata={"type": "element"})
    inertia: Inertia = field(default_factory=Inertia(), metadata={"type": "element"})


@dataclass
class GeometryChoice(UrdfXmlElement):
    """Used as a base class to the geometry elements to group them together."""

    pass


@dataclass
class Box(GeometryChoice):
    """Define the box XML element."""

    size: Vector = field(default_factory=lambda: [0, 0, 0], metadata={"type": "attribute"})

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Box) and vector_eq(self.size, other.size)


@dataclass
class Cylinder(GeometryChoice):
    """Define the cylinder XML element."""

    radius: float = field(metadata={"type": "attribute"})
    length: float = field(metadata={"type": "attribute"})


@dataclass
class Sphere(GeometryChoice):
    """Define the sphere XML element."""

    radius: float = field(metadata={"type": "attribute"})


@dataclass
class Mesh(GeometryChoice):
    """Define the mesh XML element."""

    filename: str = field(metadata={"type": "attribute"})
    scale: Vector = field(default_factory=lambda: [1, 1, 1], metadata={"type": "attribute"})

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Mesh)
            and self.filename == other.filename
            and vector_eq(self.scale, other.scale)
        )


@dataclass
class Geometry(UrdfXmlElement):
    """Define the geometry XML element."""

    choice: GeometryChoice = field(metadata={"type": "element", "name": None})


@dataclass
class Texture(UrdfXmlElement):
    """Define the texture XML element."""

    filename: str = field(metadata={"type": "attribute"})


@dataclass
class Material(UrdfXmlElement):
    """Define the material XML element."""

    name: str = field(metadata={"type": "attribute"})
    color: Optional[Color] = field(default=None, metadata={"type": "element"})
    texture: Optional[Texture] = field(default=None, metadata={"type": "element"})


@dataclass
class MaterialGlobal(UrdfXmlElement):
    """Define the material global XML element."""

    name: str = field(metadata={"type": "attribute"})
    color: Optional[Color] = field(default=None, metadata={"type": "element"})
    texture: Optional[Texture] = field(default=None, metadata={"type": "element"})


@dataclass
class Visual(UrdfXmlElement):
    """Define the visual XML element."""

    origin: Pose = field(default_factory=Pose(), metadata={"type": "element"})
    geometry: Optional[Geometry] = field(
        default=None, metadata={"type": "element", "required": True}
    )
    material: Optional[Material] = field(default=None, metadata={"type": "element"})


@dataclass
class Collision(UrdfXmlElement):
    """Define the collision XML element."""

    origin: Pose = field(default_factory=Pose(), metadata={"type": "element"})
    geometry: Optional[Geometry] = field(
        default=None, metadata={"type": "element", "required": True}
    )
    verbose: Optional[Verbose] = field(default=None, metadata={"type": "element"})


@dataclass
class Link(UrdfXmlElement):
    """Define the link XML element."""

    inertial: Optional[Inertial] = field(default=None, metadata={"type": "element"})
    visuals: List[Visual] = field(
        default_factory=list, metadata={"type": "element", "name": "visual"}
    )
    collisions: List[Collision] = field(
        default_factory=list, metadata={"type": "element", "name": "collision"}
    )
    name: Optional[str] = field(default=None, metadata={"type": "attribute", "required": True})


@dataclass
class Parent(UrdfXmlElement):
    """Define the parent XML element."""

    link: str = field(metadata={"type": "attribute"})


@dataclass
class Child(UrdfXmlElement):
    """Define the child XML element."""

    link: str = field(metadata={"type": "attribute"})


@dataclass
class Axis(UrdfXmlElement):
    """Define the axis XML element."""

    xyz: Vector = field(default_factory=lambda: [1, 0, 0], metadata={"type": "attribute"})

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Axis) and vector_eq(self.xyz, other.xyz)


@dataclass
class Calibration(UrdfXmlElement):
    """Define the calibration XML element."""

    reference_position: float = field(metadata={"type": "attribute"})
    rising: float = field(metadata={"type": "attribute"})
    falling: float = field(metadata={"type": "attribute"})


@dataclass
class Dynamics(UrdfXmlElement):
    """Define the dynamics XML element."""

    damping: float = field(default=0, metadata={"type": "attribute"})
    friction: float = field(default=0, metadata={"type": "attribute"})


@dataclass
class Limit(UrdfXmlElement):
    """Define the limit XML element."""

    lower: float = field(default=0, metadata={"type": "attribute"})
    upper: float = field(default=0, metadata={"type": "attribute"})
    effort: float = field(default=0, metadata={"type": "attribute"})
    velocity: float = field(default=0, metadata={"type": "attribute"})


@dataclass
class SafetyController(UrdfXmlElement):
    """Define the safety controller XML element."""

    lower_limit: float = field(default=0, metadata={"type": "attribute"})
    upper_limit: float = field(default=0, metadata={"type": "attribute"})
    k_position: float = field(default=0, metadata={"type": "attribute"})
    k_velocity: Optional[float] = field(
        default=None, metadata={"type": "attribute", "required": True}
    )


@dataclass
class Mimic(UrdfXmlElement):
    """Define the mimic XML element."""

    joint: str = field(metadata={"type": "attribute"})
    multiplier: float = field(default=1, metadata={"type": "attribute"})
    offset: float = field(default=0, metadata={"type": "attribute"})


@dataclass
class ActuatorTransmission(UrdfXmlElement):
    """Define the actuator transmission XML element."""

    mechanical_reduction: float = field(metadata={"type": "attribute"})
    name: str = field(metadata={"type": "attribute"})


@dataclass
class GapJointTransmission(UrdfXmlElement):
    """Define the gap joint transmission XML element."""

    L0: float = field(metadata={"type": "attribute"})
    a: float = field(metadata={"type": "attribute"})
    b: float = field(metadata={"type": "attribute"})
    gear_ratio: float = field(metadata={"type": "attribute"})
    h: float = field(metadata={"type": "attribute"})
    mechanical_reduction: float = field(metadata={"type": "attribute"})
    name: str = field(metadata={"type": "attribute"})
    phi0: float = field(metadata={"type": "attribute"})
    r: float = field(metadata={"type": "attribute"})
    screw_reduction: float = field(metadata={"type": "attribute"})
    t0: float = field(metadata={"type": "attribute"})
    theta0: float = field(metadata={"type": "attribute"})


@dataclass
class PassiveJointTransmission(UrdfXmlElement):
    """Define the passive joint transmission XML element."""

    name: str = field(metadata={"type": "attribute"})


@dataclass
class Transmission(UrdfXmlElement):
    """Define the transmission XML element."""

    left_actuator: Optional[ActuatorTransmission] = field(
        default=None, metadata={"type": "element"}
    )
    right_actuator: Optional[ActuatorTransmission] = field(
        default=None, metadata={"type": "element"}
    )
    flex_joint: Optional[ActuatorTransmission] = field(default=None, metadata={"type": "element"})
    roll_joint: Optional[ActuatorTransmission] = field(default=None, metadata={"type": "element"})
    gap_joint: Optional[GapJointTransmission] = field(default=None, metadata={"type": "element"})
    passive_joint: Optional[PassiveJointTransmission] = field(
        default=None, metadata={"type": "element"}
    )
    mechanical_reduction: Optional[float] = field(default=None, metadata={"type": "element"})
    actuator: Optional[Name] = field(default=None, metadata={"type": "element"})
    joint: Optional[Name] = field(default=None, metadata={"type": "element"})
    name: Optional[str] = field(default=None, metadata={"type": "attribute", "required": True})
    type: Optional[str] = field(default=None, metadata={"type": "attribute", "required": True})


@dataclass
class Joint(UrdfXmlElement):
    """Define the joint XML element."""

    origin: Optional[Pose] = field(default=None, metadata={"type": "element"})
    parent: Optional[Parent] = field(default=None, metadata={"type": "element", "required": True})
    child: Optional[Child] = field(default=None, metadata={"type": "element", "required": True})
    axis: Optional[Axis] = field(default=None, metadata={"type": "element"})
    calibration: Optional[Calibration] = field(default=None, metadata={"type": "element"})
    dynamics: Optional[Dynamics] = field(default=None, metadata={"type": "element"})
    limit: Optional[Limit] = field(default=None, metadata={"type": "element"})
    safety_controller: Optional[SafetyController] = field(
        default=None, metadata={"type": "element"}
    )
    mimic: Optional[Mimic] = field(default=None, metadata={"type": "element"})
    name: Optional[str] = field(default=None, metadata={"type": "attribute", "required": True})
    type: Optional[str] = field(default=None, metadata={"type": "attribute", "required": True})


@dataclass
class Robot(UrdfXmlElement):
    """Define the robot XML element."""

    joints: List[Joint] = field(default_factory=list, metadata={"type": "element", "name": "joint"})
    links: List[Link] = field(default_factory=list, metadata={"type": "element", "name": "link"})
    materials: List[MaterialGlobal] = field(
        default_factory=list, metadata={"type": "element", "name": "material"}
    )
    transmission: List[Transmission] = field(
        default_factory=list, metadata={"type": "element", "name": "transmission"}
    )
    name: Optional[str] = field(default=None, metadata={"type": "attribute", "required": True})
