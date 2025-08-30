# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""SRL world model module."""


# Standard Library
import functools
import os
import pathlib
from typing import Any, List, Optional, Tuple, Union

# Third Party
from pxr import Sdf, Usd, UsdGeom, UsdPhysics
from tqdm import tqdm

# NVIDIA
from lightwheel.srl.abc.srl import SRL
from lightwheel.srl.basics.types import FrameState, PathLike, Pose, WorldState
from lightwheel.srl.math.transform import Transform
from lightwheel.srl.usd import prim_helper


class WorldStructure(SRL):
    """Class to manage the world structure."""

    DEFAULT_URDF_REL_DIR = "./urdf"

    def __init__(self, stage: Usd.Stage, urdf_rel_dir_path: Optional[str] = None, **kwargs: Any):
        """Initialize a new `WorldModel` object.

        Args:
            stage: USD stage that describes the world scene.
            urdf_rel_dir_path: Relative path from the USD file's parent directory to the URDF
                directory.
            kwargs: Additional keyword arguments are passed to the parent class,
                :class:`~srl.abc.srl.SRL`.
        """
        # Initialize parent class
        super().__init__(**kwargs)

        # Initialize instance attributes
        if urdf_rel_dir_path is None:
            urdf_rel_dir_path = self.DEFAULT_URDF_REL_DIR
        self._urdf_rel_dir_path = urdf_rel_dir_path

        self.stage = stage

    @classmethod
    def init_from_usd_path(cls, usd_path: PathLike, **kwargs: Any) -> "WorldStructure":
        """Create new `WorldStructure` object from USD path.

        Args:
            usd_path: File path to load the USD from.

        Returns:
            WorldStructure: New `WorldStructure` object initialized from USD path.
        """
        stage = prim_helper.open_stage(usd_path)
        return cls(stage, **kwargs)

    def get_world_usd_path(self) -> str:
        """Get path to the world USD file."""
        return self.stage.GetRootLayer().realPath

    def get_urdf_root_dir_path(self) -> str:
        """Get the URDF root directory path."""
        world_usd_path = pathlib.Path(self.get_world_usd_path()).resolve()
        urdf_root_dir_path = world_usd_path.parent / self._urdf_rel_dir_path
        return str(urdf_root_dir_path)

    def get_urdf_path(self, prim: Usd.Prim) -> str:
        """Get expected URDF absolute path for the given prim."""
        default_prim_path = self.stage.GetDefaultPrim().GetPath().pathString + "/"
        rel_prim_path = prim.GetPath().pathString.split(default_prim_path)[-1]
        rel_urdf_path = rel_prim_path + ".urdf"
        return str(pathlib.Path(self.get_urdf_root_dir_path()) / rel_urdf_path)

    def get_world_prim(self) -> Usd.Prim:
        """Get the default prim in the stage."""
        return prim_helper.get_world_prim(self.stage)

    def get_entity_prims(self) -> List[Usd.Prim]:
        """Get a list of entity prims that are children of the world prim."""
        return list(filter(prim_helper.is_entity, self.get_world_prim().GetChildren()))

    def get_object_prims(self) -> List[Usd.Prim]:
        """Get a list of object prims that are children of the world prim."""
        return list(filter(prim_helper.is_object, self.get_world_prim().GetChildren()))

    def get_robot_prims(self) -> List[Usd.Prim]:
        """Get a list of robot prims that are children of the world prim."""
        return list(filter(prim_helper.is_robot, self.get_world_prim().GetChildren()))

    def get_camera_prims(self) -> List[Usd.Prim]:
        """Get a list of camera prims that are descendants of the world prim."""
        return list(
            filter(prim_helper.is_camera, prim_helper.get_descendant_prims(self.get_world_prim()))
        )

    def get_initial_state_dict(self) -> WorldState:
        """Get the initial world state dictionary."""
        world_state_dict = {}

        # Add all collision objects that are in the environment
        for prim in self.get_entity_prims():
            world_state_dict[prim.GetPath().pathString] = prim_helper.get_state(prim)

        return world_state_dict

    def _create_stage(
        self,
        start_time_code: Optional[float] = None,
        end_time_code: Optional[float] = None,
        time_codes_per_second: Optional[float] = None,
    ) -> Usd.Stage:
        # Configure parameters
        if start_time_code is None:
            start_time_code = self.stage.GetStartTimeCode()
        if end_time_code is None:
            end_time_code = self.stage.GetEndTimeCode()
        if time_codes_per_second is None:
            time_codes_per_second = self.stage.GetTimeCodesPerSecond()

        # Create stage
        stage = Usd.Stage.CreateInMemory()

        # Set stage level metadata
        # TODO (roflaherty): Figure out if there is a better way to do this
        stage.SetDefaultPrim(self.stage.GetDefaultPrim())
        stage.SetStartTimeCode(start_time_code)
        stage.SetEndTimeCode(end_time_code)
        stage.SetTimeCodesPerSecond(time_codes_per_second)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(self.stage))
        UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(self.stage))
        UsdPhysics.SetStageKilogramsPerUnit(stage, UsdPhysics.GetStageKilogramsPerUnit(self.stage))

        return stage

    def to_usd_file(
        self,
        states: Union[WorldState, List[WorldState]],
        output_file: str,
        seconds_per_state: Optional[float] = None,
        show_progress_bar: bool = True,
        usd_ref_rel_path: bool = False,
    ) -> None:
        """Create a USD file for the given set of states.

        Note:
            If `states` is a single `WorldState` then this will create just an overlay stage for
            that single state. If `states` is a list of `WorldState` then this will create an
            overlay stage that is an animation of the objects in the world moving through their
            state values.

        Args:
            states: A single state or list states defined as WorldStructure state dictionaries.
            output_file: Output file where the USD will be saved to.
            seconds_per_state: Sets the time duration for each state (i.e. inverse of frame
                rate, set by the USD property `time_codes_per_second`).
            show_progress_bar: If true, a progress bar is shown.
            usd_ref_rel_path: If true, the relative paths will be used in the USD for referencing
                layers.
        """
        pathlib.Path(output_file)
        stage = self.to_usd_stage(states, seconds_per_state, show_progress_bar)
        if usd_ref_rel_path:
            rel_world_usd_path = os.path.relpath(
                self.get_world_usd_path(), pathlib.Path(output_file).parent
            )
            refs = stage.GetPrimAtPath("/world").GetReferences()
            refs.ClearReferences()
            refs.AddReference(rel_world_usd_path)
        usd_str = stage.GetRootLayer().ExportToString()
        with open(output_file, "w") as file:
            file.write(usd_str)

    def to_usd_str(
        self,
        states: Union[WorldState, List[WorldState]],
        seconds_per_state: Optional[float] = None,
        show_progress_bar: bool = True,
    ) -> str:
        """Create a USD string for the given set of states.

        Note:
            If `states` is a single `WorldState` then this will create just an overlay stage for
            that single state. If `states` is a list of `WorldState` then this will create an
            overlay stage that is an animation of the objects in the world moving through their
            state values.

        Args:
            states: A single state or list states defined as WorldStructure state dictionaries.
            seconds_per_state: Sets the time duration for each state (i.e. inverse of frame
                rate, set by the USD property `time_codes_per_second`).
            show_progress_bar: If true, a progress bar is shown.

        Returns:
            USD as string with the world structure USD overridden with the state transform
            information.
        """
        stage = self.to_usd_stage(states, seconds_per_state, show_progress_bar)
        return stage.GetRootLayer().ExportToString()

    @functools.lru_cache(maxsize=None)
    def _lookup_frame_name(
        self, entity_prim: Usd.Prim, frame_prim: Usd.Prim, frame_names: Tuple[str]
    ) -> Optional[str]:
        """Lookup a `FrameState` frame name for a given frame prim."""
        entity_path = prim_helper.get_prim_path(entity_prim)
        frame_path = prim_helper.get_prim_path(frame_prim)
        frame_name = os.path.relpath(frame_path, entity_path).replace("/", "_")
        if frame_name in frame_names:
            return frame_name
        for _frame_name in frame_names:
            if frame_name.endswith(_frame_name):
                self.logger.warning(
                    f"USD frame '{frame_name}' does not exactly map to a state frame. Using state"
                    f" frame with common suffix '{_frame_name}'."
                )
                return _frame_name
        return None

    def _extract_frame_pose(
        self, entity_prim: Usd.Prim, frame_prim: Usd.Prim, frame_state: FrameState
    ) -> Pose:
        """Extract a `FrameState` frame pose for a given frame prim."""
        frame_names = tuple(frame_state.get("name", []))
        if len(frame_names) == 0:
            raise ValueError(f"Frame state `name` empty for entity {entity_prim}.")
        frame_poses = frame_state.get("pose", [])
        if len(frame_names) != len(frame_poses):
            raise ValueError(
                f"Frame state `name` and `pose` for entity {entity_prim} differ in"
                f" length ({len(frame_names)} versus {len(frame_poses)})."
            )
        frame_name = self._lookup_frame_name(entity_prim, frame_prim, frame_names)
        if frame_name is None:
            raise ValueError(f"Frame '{frame_prim}' is not listed in state frames: {frame_names}.")
        frame_idx = frame_names.index(frame_name)
        return frame_poses[frame_idx]

    def to_usd_stage(
        self,
        states: Union[WorldState, List[WorldState]],
        seconds_per_state: Optional[float] = None,
        show_progress_bar: bool = True,
    ) -> Usd.Stage:
        """Create a new USD stage for the given set of states.

        Note:
            If `states` is a single `WorldState` then this will create just an overlay stage for
            that single state. If `states` is a list of `WorldState` then this will create an
            overlay stage that is an animation of the objects in the world moving through their
            state values.

        Args:
            states: A single state or list states defined as WorldStructure state dictionaries.
            seconds_per_state: Sets the time duration for each state (i.e. inverse of frame
                rate, set by the USD property `time_codes_per_second`).
            show_progress_bar: If true, a progress bar is shown.

        Returns:
            USD stage with the world structure USD overridden with the state transform
            information.
        """
        # Set default arguments
        if not isinstance(states, list):
            states = [states]
        if (seconds_per_state is None) or (seconds_per_state == 0):
            time_codes_per_second = None
        else:
            time_codes_per_second = 1 / seconds_per_state

        # Create stage
        start_time_code = 0
        end_time_code = len(states) - 1
        stage = self._create_stage(start_time_code, end_time_code, time_codes_per_second)

        # Override world prim
        world_prim = stage.OverridePrim("/world")
        world_prim.GetReferences().AddReference(self.get_world_usd_path())

        for prim in stage.Traverse():
            if prim.IsInstanceable():
                # Disable instances to mutate prim poses
                prim.SetInstanceable(False)
            attribute = prim.GetAttribute("physics:rigidBodyEnabled")
            if attribute.IsValid():
                attribute.Set(False)

        # NOTE (cgarrett): This can be simplified to `@functools.cache` in Python >=3.9.
        # See https://docs.python.org/3.9/library/functools.html#functools.cache
        @functools.lru_cache(maxsize=None)
        def get_entity_joints(_entity_prim: Usd.Prim) -> List[Usd.Prim]:
            return prim_helper.get_joints_for_articulated_root(_entity_prim)

        @functools.lru_cache(maxsize=None)
        def get_entity_links(_entity_prim: Usd.Prim) -> List[Usd.Prim]:
            return prim_helper.get_links_for_articulated_root(_entity_prim)

        @functools.lru_cache(maxsize=None)
        def get_entity_movable_links(_entity_prim: Usd.Prim) -> List[Usd.Prim]:
            return prim_helper.get_links_for_articulated_root(
                _entity_prim, prim_helper.is_a_movable_joint
            )

        # Loop states
        disable_progress_bar = not show_progress_bar
        progress_bar_desc = "Percent of USD animation complete"
        for idx, state in enumerate(
            tqdm(states, desc=progress_bar_desc, leave=False, disable=disable_progress_bar)
        ):
            # Loop through state items
            for prim_path, state_entity in state.items():
                # Get the state entity's prim object
                entity_prim = stage.GetPrimAtPath(prim_path)

                # Set the entity's pose if it is floating
                if state_entity["pose_type"] == "PoseType.FLOATING":
                    entity_pose = state_entity["pose"]
                    prim_helper.set_transform_local(entity_prim, entity_pose, idx)

                # Set the entity's joint state if it is articulated
                if (
                    state_entity["body_type"] == "BodyType.ARTICULATED"
                ) and prim_helper.is_articulated_root(entity_prim):
                    for joint_prim in get_entity_joints(entity_prim):
                        joint_prim.GetAttribute("physics:jointEnabled").Set(False)

                    # Make all links kinematic
                    for link_prim in get_entity_movable_links(entity_prim):
                        if not link_prim.GetAttribute("physics:kinematicEnabled").IsValid():
                            link_prim.CreateAttribute(
                                "physics:kinematicEnabled", Sdf.ValueTypeNames.Bool
                            )
                        link_prim.GetAttribute("physics:kinematicEnabled").Set(True)

                        if not link_prim.GetAttribute("physics:rigidBodyEnabled").IsValid():
                            link_prim.CreateAttribute(
                                "physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool
                            )
                        link_prim.GetAttribute("physics:rigidBodyEnabled").Set(False)

                    # Set `joint_positions` property
                    joint_state = state_entity.get("joint_state", {})
                    if not joint_state:
                        raise ValueError(
                            f"State dict `joint_state` is empty for entity {prim_path}."
                        )
                    prim_helper.set_joint_state(entity_prim, joint_state)

                    frame_state = state_entity.get("frame_state", {})
                    if not frame_state:
                        raise ValueError(f"State dict `frame_state` empty for entity {prim_path}.")
                    frame_names = tuple(frame_state.get("name", []))

                    # Set the link frame's pose
                    for frame_prim in get_entity_links(entity_prim):
                        if frame_prim == entity_prim:
                            # Entity prim pose has already been set
                            continue
                        if self._lookup_frame_name(entity_prim, frame_prim, frame_names) is None:
                            # No new pose specified for frame prim
                            continue
                        frame_pose = self._extract_frame_pose(entity_prim, frame_prim, frame_state)
                        parent_prim = frame_prim.GetParent()
                        if parent_prim == entity_prim:
                            # Parent is entity prim
                            parent_pose = state_entity["pose"]
                        elif self._lookup_frame_name(entity_prim, parent_prim, frame_names) is None:
                            # No new pose specified for parent prim
                            parent_pose = prim_helper.get_pose(parent_prim, idx)
                        else:
                            parent_pose = self._extract_frame_pose(
                                entity_prim, parent_prim, frame_state
                            )
                        relative_pose = Transform.inverse(parent_pose) @ frame_pose
                        prim_helper.set_transform_local(frame_prim, relative_pose, idx)

        return stage
