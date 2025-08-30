# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Functions to convert from USD."""

# Standard Library
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os

# Third Party
import numpy as np
from pxr import Usd, UsdGeom, UsdShade, UsdPhysics

# NVIDIA
from lightwheel.srl.mojoco_xml import elements as mjcf
import lightwheel.srl.usd.prim_helper as prim_helper
from lightwheel.srl.abc.srl import SRL
from lightwheel.srl.basics.types import PathLike
from lightwheel.srl.from_usd._from_usd_helper import NodeType, _get_prim_scale,UsdMaterialWrapper
from lightwheel.srl.from_usd.transform_graph import TransformEdge, TransformGraph, TransformNode
from lightwheel.srl.from_usd.transform_graph_tools import reduce_to_mjcf
from lightwheel.srl.math.transform import Transform
from lightwheel.srl.from_usd.usd_to_obj import export_mesh_subset_to_obj, export_mesh_to_obj

class UsdToMjcf(SRL):
    """Class used to convert USD files to MJCF files."""

    def __init__(
        self,
        stage: Usd.Stage,
        node_names_to_remove: Optional[str] = None,
        edge_names_to_remove: Optional[str] = None,
        root: Optional[str] = None,
        parent_link_is_body_1: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """Initialize a new `UsdToMjcf` object.

        Args:
            stage: USD stage that describes the scene.
            node_names_to_remove: List of node names to remove from the `TransformGraph` to break
                kinematic loops and make the graph transformable to something valid to create a MJCF.
            edge_names_to_remove: List of edge names to remove from the `TransformGraph` to break
                kinematic loops and make the graph transformable to something valid to create a MJCF.
            root: The root node name that will be set as the root of the kinematic structure of the
                new MJCF.  This sets the "robot" element in the new MJCF. The root node can either
                be specified with the prim path or with the node name.
            parent_link_is_body_1: A list of joint node names where the parent link is assumed to be
                the body 1 target prim, instead of the default body 0 target prim. Note, when only
                one body target is set, then the parent link is assumed to be the default prim, and
                the child link is the prim of whatever body target that is set.
            kinematics_only: If true, the re
            kwargs: Additional keyword arguments are passed to the parent class,
                `SRL <https://srl.gitlab-master-pages.nvidia.com/py/base/_api/nvidia.srl.abc.srl.html#nvidia.srl.abc.srl.SRL>`_.
        """  # noqa: E501
        # Initialize parent class
        super().__init__(**kwargs)

        # Initialize instance variables
        self._usd_path: Optional[Path] = None
        self._stage = stage
        log_level = kwargs.get("log_level", None)
        self._graph = TransformGraph.init_from_stage(
            stage,
            parent_link_is_body_1=parent_link_is_body_1,
            log_level=log_level,
        )
        self.exported_mjcf_materials = set()
        nodes_to_remove: Optional[List[TransformNode]] = None
        if node_names_to_remove is not None:
            nodes_to_remove = list(
                map(lambda name_: self._graph.get_node(name_), node_names_to_remove)
            )

        edges_to_remove: Optional[List[TransformEdge]] = None
        if edge_names_to_remove is not None:
            edges_to_remove = list(
                map(lambda name_: self._graph.get_edge(name_), edge_names_to_remove)
            )

        if root is None:
            root_node = None
        else:
            root_node = self._graph.get_node(root)

        reduce_to_mjcf(
            self._graph,
            nodes_to_remove=nodes_to_remove,
            edges_to_remove=edges_to_remove,
            root_node=root_node,
        )

        if not self._graph.is_possible_mjcf():
            msg = (
                "Cannot build MJCF from USDs that are not structured as kinematic trees. Consider"
                " restructuring your USD stage, or using the `node_names_to_remove`,"
                " `edge_names_to_remove`, and/or `parent_link_is_body_1` options to make the the"
                " transform graph into a tree."
            )
            raise RuntimeError(msg)

    @classmethod
    def init_from_file(cls, usd_path: PathLike,**kwargs: Any) -> "UsdToMjcf":
        """Create a new `UsdToMjcf` object from a USD file.

        Args:
            usd_path: Path to USD file that describes the scene.
            kwargs: Additional keyword arguments are passed to
                 :class:`UsdToMjcf.__init__()<UsdToMjcf>`.

        Returns:
            UsdToMjcf: New `UsdToMjcf` object initialized from USD path.
        """
        if not isinstance(usd_path, str):
            usd_path = str(usd_path)
        stage = Usd.Stage.Open(usd_path)
        usd_to_mjcf = cls(stage, **kwargs)
        usd_to_mjcf._usd_path = Path(usd_path)
        return usd_to_mjcf

    def save_to_file(self, mjcf_output_path: PathLike, quiet: bool = False, **kwargs: Any) -> Path:
        """Convert the USD to a MJCF and save to file.

        Args:
            mjcf_output_path: The path to where the MJCF file will be saved. If it is a file path
                then it is saved to that file. If it is a directory path, then it is a saved into
                that directory with the file name matching the USD name but with the .mjcf
                extension. If the pathe doesn't exist, then file paths are assumed to have
                extensions (usually the ".mjcf" extension) and directory paths are assumed to not
                have extensions.
            quiet: If true, nothing is printed or written to the logs.
        """
        if not isinstance(mjcf_output_path, Path):
            mjcf_output_path = Path(mjcf_output_path)

        if (
            mjcf_output_path.exists() and mjcf_output_path.is_file()
        ) or mjcf_output_path.suffix != "":
            # `mjcf_output_path` assummed to be a file
            output_file = mjcf_output_path
            output_dir = mjcf_output_path.parent
        elif (
            mjcf_output_path.exists() and mjcf_output_path.is_dir()
        ) or mjcf_output_path.suffix == "":
            # `mjcf_output_path` assummed to be a directory
            output_dir = mjcf_output_path
            if self._usd_path is None:
                output_name = self._graph.name + ".xml"
            else:
                output_name = self._usd_path.stem + ".xml"
            output_file = output_dir / output_name

        else:
            msg = (
                f"The MJCF output is not valid: {mjcf_output_path}. It must be either a path to a"
                " file or a directory."
            )
            ValueError(msg)

        output_dir.mkdir(parents=True, exist_ok=True)

        #self.save_graphviz(output_dir=output_dir.as_posix(), name=self._graph.name)
        mjcf_str = self.to_str(output_dir=output_dir.as_posix(), quiet=True, **kwargs)

        with open(output_file.as_posix(), "w") as file:
            file.write(mjcf_str)

        # Log result
        if not quiet:
            if self._usd_path is None:
                usd_path = self._stage.GetRootLayer().realPath
            else:
                usd_path = self._usd_path

            msg = "\n".join(
                [
                    "Converted USD to MJCF.",
                    f"    Input file: {usd_path}",
                    f"    Output file: {output_file}",
                ]
            )
            self.logger.info(msg)

        return output_file

    def to_str(
        self,
        output_dir: Optional[PathLike] = None,
        mesh_dir: Optional[PathLike] = None,
        mesh_path_prefix: str = "",
        visualize_collision_meshes: bool = False,
        kinematics_only: bool = False,
        quiet: bool = False,
    ) -> str:
        """Convert the USD to MJCF and return the MJCF XML string.

        Args:
            output_dir: The directory to where the MJCF file would be saved to. This is used to
                calculate the relative path between the mesh directory and the output directory.
            mesh_dir: The directory where the mesh files will be stored. Defaults to creating a
                "visuals" directory in the output directory if that is set, otherwise in the current
                working directory.
            mesh_path_prefix: Set the prefix to use for the MJCF mesh filename. For example, to use
                an absolute path set this to '$(pwd)/'. Or to use a URI with the 'file' scheme,
                then set this to 'file://'.
            visualize_collision_meshes: If true, the collision meshes will be added to the set of
                visual elements in the link elements.
            kinematics_only: If true, the resulting MJCF will not contain any visual or collision
                mesh information.
            quiet: If true, nothing is printed or written to the logs.

        Returns:
            The MJCF XML string.
        """
        if output_dir is None:
            output_dir = Path.cwd().as_posix()

        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        if isinstance(mesh_dir, str):
            mesh_dir = Path(mesh_dir)

        if mesh_dir is None:
            mesh_dir = output_dir / "visuals"
        self._mesh_dir_path = mesh_dir.expanduser().resolve()
        self._material_file_path = self._mesh_dir_path / "materials.mtl"
        self._output_dir_path = output_dir
        self._mesh_path_prefix = mesh_path_prefix
        self._visualize_collision_meshes = visualize_collision_meshes
        self._kinematics_only = kinematics_only

        mjcf_str = self._build_mjcf()

        # Log result
        if not quiet:
            if self._usd_path is None:
                usd_path = self._stage.GetRootLayer().realPath
            else:
                usd_path = self._usd_path
            msg = "\n".join(
                [
                    "Converted USD to MJCF.",
                    f"    Input file: {usd_path}",
                ]
            )
            self.logger.info(msg)

        return mjcf_str

    def save_graphviz(
        self,
        output_dir: Optional[PathLike] = None,
        name: Optional[str] = None,
    ) -> Tuple[Path, Path]:
        """Save the `TransformGraph` of the USD as a graphviz dot file and rendered PNG file.

        Args:
            output_dir: The directory to where the graphviz files will be saved to. Defaults to the
            current working directory.

        Returns:
            dot_file_path: Path to the generated dot file.
            png_file_path: Path to the generated png file.
        """
        if output_dir is None:
            output_dir = Path.cwd().as_posix()

        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        dot_file_path = self._graph.save_graphviz(output_dir=output_dir, name=name)
        png_file_path = self._graph.render_graphviz(output_dir=output_dir, name=name)

        return dot_file_path, png_file_path

    def _build_mjcf(self):
        mjcf_name = self._graph.name
        mujoco = self._init_mjcf(mjcf_name)
        
        for root_node in self._graph.get_roots():
            if root_node.type == NodeType.LINK:
                root_body = self._add_body(self.worldbody,root_node)
                self._add_subtree(root_body,root_node)
      
        model_xml = mujoco.xml()
        return model_xml
    
    def _init_mjcf(self,name):
        
        mujoco = mjcf.Mujoco(
            model=name
        )
        
        compiler = mjcf.Compiler(
            angle="radian",
            coordinate="local",
            inertiafromgeom=True
        )
        option = mjcf.Option(
            integrator="RK4",
            timestep=0.01
        )
        custom = mjcf.Custom()
        self.asset = mjcf.Asset()
        self.worldbody = mjcf.Worldbody()
        self.actuator = mjcf.Actuator()
        self.default = mjcf.Default()
        
        #visual class
        visual_default = mjcf.Default(
            class_="visual"
        )
        visual_geom = mjcf.Geom(
            conaffinity="0",
            contype="0",
            group="1",
            type="mesh"
        )
        visual_default.add_child(visual_geom)
        self.default.add_child(visual_default)
        
        #collision class
        collision_default = mjcf.Default(
            class_="collision"
        )
        collision_geom = mjcf.Geom(
            group="0"
            # rgba="0 0 0 1"
        )
        collision_default.add_child(collision_geom)
        self.default.add_child(collision_default)
        
        mujoco.add_children([
            compiler,
            option,
            custom,
            self.asset,
            self.worldbody,
            self.actuator,
            self.default
        ])
        
        # default light
        test_light = mjcf.Light(
            cutoff=100,
            diffuse=[1, 1, 1],
            dir=[-0, 0, -1.3],
            directional=True,
            exponent=1,
            pos=[0, 0, 1.3],
            specular=[.1, .1, .1]
        )
        self.worldbody.add_children([
            test_light
        ])

        return mujoco
    
    def _add_geometry(self, parent_element: mjcf.Element,node: TransformNode) -> None:
            # A geometry node should never be the root node.
            assert not node.is_root

            # NOTE (roflaherty): This is a hack to deal with the hidden camera geometry prims that only
            # exist when converting from an open stage in Isaac Sim using the USD to MJCF Exporter
            # extension.
            if "CameraModel" in node.name:
                msg = f"Skipping adding {node.name} geometry."
                self.logger.debug(msg)
                return

            if not self._kinematics_only:
                node_parent = node.to_neighbors[0]
                geometry_visuals = self._get_geometry(node)
                for geometry_visual in geometry_visuals:
                    parent_element.add_child(geometry_visual)

                if not node.is_leaf or (
                    node_parent.type != NodeType.LINK and node_parent.type != NodeType.PHONY
                ):
                    msg = (
                        f"The '{node.path}' geometry is not a leaf node or is connected to a joint"
                        " directly. The USD to MJCF convertered currently only supports converting"
                        " USDs where geometries are leaf nodes. This means that geometries (i.e."
                        " meshes and primitives prims) must be child prims of Xform prims and must not"
                        " connect to joint prims directly."
                    )
                    raise RuntimeError(msg)
            return

    def _need_collision_filter(self, prim):
        return prim.HasAPI(UsdPhysics.CollisionAPI)

    def _get_geometry(
        self, node: TransformNode
    ) -> List[mjcf.Geom]:
        #Todo: support omni complex inherited visibility properties
        #Todo: support omni complex collision preset properties
        
        """Get mjcf.Geom objects for the prim.

        Args:
            node: The node to get the geometry for

        Returns:
            visual geometry: The visual geometry for the prim.
            collision geometry: The collision geometry for the prim.
            origin: TBD.
        """
        if node.prim is None:
            raise RuntimeError(
                "Something is wrong. A geometry node shoud always have a prim value."
            )
        prim = node.prim
        geometry_visuals: List[mjcf.Geom] = []
        need_filter = self._need_collision_filter(prim)
        if need_filter:
            return geometry_visuals
           
        UsdGeom.Imageable(prim).GetVisibilityAttr().Get()
        
        link_t_geometry = node.from_edges[0].transform
        prim_scale = _get_prim_scale(prim)
        transform_scale = Transform.get_scale(link_t_geometry)

        if prim_scale is not None and np.allclose(transform_scale, np.ones(3)):
            scale = prim_scale
        else:
            scale = transform_scale

        
        #geometry_collision: Optional[mjcf.Geom] = None

        rotation_correction = Transform.identity()

        if prim.IsA(UsdGeom.Cylinder):
            radius = prim.GetAttribute("radius").Get()
            length = prim.GetAttribute("height").Get()
            axis = prim.GetAttribute("axis").Get()
            if axis == "X":
                rotation_correction = Transform.from_rotvec(np.array([0, np.pi / 2, 0]))
                scale_r1 = scale[1]
                scale_r2 = scale[2]
                scale_h = scale[0]
            elif axis == "Y":
                rotation_correction = Transform.from_rotvec(np.array([np.pi / 2, 0, 0]))
                scale_r1 = scale[0]
                scale_r2 = scale[2]
                scale_h = scale[1]
            elif axis == "Z":
                rotation_correction = Transform.identity()
                scale_r1 = scale[0]
                scale_r2 = scale[1]
                scale_h = scale[2]
            else:
                raise RuntimeError(f"Unknown cylinder axis value '{axis}' for '{node.path}'.")

            if not np.isclose(scale_r1, scale_r2):
                msg = (
                    "MJCF cannot scale radii of each axis of a cylinder different amounts."
                    f" Cylinder '{node.path}' has scaling factors of {scale_r1} and {scale_r2} for"
                    " non-height axes."
                )
                raise RuntimeError(msg)
            origin,quat_wxyz = UsdToMjcf._get_mjcf_pose(link_t_geometry @ rotation_correction)
            geometry_visual = mjcf.Geom(
                name=node.name,
                pos=[float(origin[0]),float(origin[1]),float(origin[2])],
                quat=[float(quat_wxyz[0]),float(quat_wxyz[1]),float(quat_wxyz[2]),float(quat_wxyz[3])],
                size=[float(radius*scale_r1), float(length/2*scale_h)],
                type="cylinder",
                rgba=None
            )
            geometry_visuals.append(geometry_visual)
        elif prim.IsA(UsdGeom.Capsule):
            radius = prim.GetAttribute("radius").Get()
            length = prim.GetAttribute("height").Get()
            axis = prim.GetAttribute("axis").Get()
            if axis == "X":
                rotation_correction = Transform.from_rotvec(np.array([0, np.pi / 2, 0]))
                scale_r1 = scale[1]
                scale_r2 = scale[2]
                scale_h = scale[0]
            elif axis == "Y":
                rotation_correction = Transform.from_rotvec(np.array([np.pi / 2, 0, 0]))
                scale_r1 = scale[0]
                scale_r2 = scale[2]
                scale_h = scale[1]
            elif axis == "Z":
                rotation_correction = Transform.identity()
                scale_r1 = scale[0]
                scale_r2 = scale[1]
                scale_h = scale[2]
            else:
                raise RuntimeError(f"Unknown cylinder axis value '{axis}' for '{node.path}'.")

            if not np.isclose(scale_r1, scale_r2):
                msg = (
                    "MJCF cannot scale radii of each axis of a cylinder different amounts."
                    f" Cylinder '{node.path}' has scaling factors of {scale_r1} and {scale_r2} for"
                    " non-height axes."
                )
                raise RuntimeError(msg)
            origin,quat_wxyz = UsdToMjcf._get_mjcf_pose(link_t_geometry @ rotation_correction)
            geometry_visual = mjcf.Geom(
                name=node.name,
                pos=[float(origin[0]),float(origin[1]),float(origin[2])],
                quat=[float(quat_wxyz[0]),float(quat_wxyz[1]),float(quat_wxyz[2]),float(quat_wxyz[3])],
                size=[float(scale_r1), float(scale_r2)],
                type="capsule",
                rgba=None
            )
            geometry_visuals.append(geometry_visual)
        elif prim.IsA(UsdGeom.Cube):
            attr_size = prim.GetAttribute("size").Get()
            size = scale * attr_size/2
            origin,quat_wxyz = UsdToMjcf._get_mjcf_pose(link_t_geometry @ rotation_correction)
            geometry_visual = mjcf.Geom(
                name=node.name,
                pos=[float(origin[0]),float(origin[1]),float(origin[2])],
                quat=[float(quat_wxyz[0]),float(quat_wxyz[1]),float(quat_wxyz[2]),float(quat_wxyz[3])],
                size= [float(size[0]), float(size[1]), float(size[2])],
                type="box",
                rgba=None
            )
            geometry_visuals.append(geometry_visual)
        elif prim.IsA(UsdGeom.Sphere):
            radius = prim.GetAttribute("radius").Get()
            origin,quat_wxyz = UsdToMjcf._get_mjcf_pose(link_t_geometry @ rotation_correction)
            geometry_visual = mjcf.Geom(
                name=node.name,
                pos=[float(origin[0]),float(origin[1]),float(origin[2])],
                quat=[float(quat_wxyz[0]),float(quat_wxyz[1]),float(quat_wxyz[2]),float(quat_wxyz[3])],
                size=float(radius*scale[0]),
                type="sphere",
                rgba=None
            )
            geometry_visuals.append(geometry_visual)
        elif prim.IsA(UsdGeom.Mesh):
            is_collision = prim_helper.is_collider(prim)
            origin,quat_wxyz = UsdToMjcf._get_mjcf_pose(link_t_geometry @ rotation_correction)
            self._mesh_dir_path.mkdir(parents=True, exist_ok=True)
            
            # Get material subsets
            mesh = UsdGeom.Mesh(prim)
            subsets = UsdGeom.Subset.GetAllGeomSubsets(mesh)
            mesh_binding = UsdShade.MaterialBindingAPI(prim)
            mesh_material = mesh_binding.ComputeBoundMaterial()[0]
            if not subsets:
                # Export entire mesh as a single OBJ file
                mesh_filename = f"{node.name}.obj"
                obj_output_path = self._mesh_dir_path / mesh_filename
                mtl_file_path = self._material_file_path
                
                material_info = export_mesh_to_obj(mesh, obj_output_path, mesh_material,mtl_file_path)
                
                relative_obj_path = os.path.relpath(obj_output_path, self._output_dir_path)
                mjcf_mesh = mjcf.Mesh(
                    file=relative_obj_path,
                    name=node.name,
                    scale=[float(scale[0]),float(scale[1]),float(scale[2])]
                )
                self.asset.add_child(mjcf_mesh)
                geometry_visual = mjcf.Geom(
                    name=node.name,
                    type="mesh",
                    pos=[float(origin[0]),float(origin[1]),float(origin[2])],
                    quat=[float(quat_wxyz[0]),float(quat_wxyz[1]),float(quat_wxyz[2]),float(quat_wxyz[3])],
                    mesh=node.name,
                )
                mesh_material_prim =mesh_material.GetPrim()
                if mesh_material_prim:
                    material_name = mesh_material_prim.GetName()
                    geometry_visual.material = material_name
                    self._generate_mjcf_material(material_info,material_name)
                geometry_visuals.append(geometry_visual) 
                
                if is_collision:
                    geometry_collision = mjcf.Geom(
                        name=node.name,
                        type="mesh",
                        pos=[float(origin[0]),float(origin[1]),float(origin[2])],
                        quat=[float(quat_wxyz[0]),float(quat_wxyz[1]),float(quat_wxyz[2]),float(quat_wxyz[3])],
                        mesh=node.name,
                    )
                
            else:
                # Export separate OBJ for each material subset
                for index,subset in enumerate(subsets):
                    binding = UsdShade.MaterialBindingAPI(subset.GetPrim())
                    subset_material = binding.ComputeBoundMaterial()[0]
                    geom_name =f"{node.name}_{index}"
                    
                    geometry_visual = self._generate_subset_geometry(mesh,subset,subset_material,geom_name,scale,origin,quat_wxyz)
                    if geometry_visual is not None:
                        geometry_visuals.append(geometry_visual)
                    
        else:
            raise RuntimeError(
                "Invalid prim type. Valid types: `UsdGeom.Cylinder`, `UsdGeom.Capsule`,`UsdGeom.Cube`,"
                " `UsdGeom.Sphere`, `UsdGeom.Mesh`."
            )

        if prim_helper.is_collider(prim):
            for geometry_visual in geometry_visuals:
                #geometry_visual.class_ = "collision"
                geometry_visual.class_ = "visual"
        else:
            for geometry_visual in geometry_visuals:
                geometry_visual.class_ = "visual"
                
        return geometry_visuals
    
    def _generate_subset_geometry(self,mesh,subset,material,geom_name,scale,origin,quat_wxyz):
        
        obj_file_path = self._mesh_dir_path / f"{geom_name}.obj"
        mtl_file_path = self._material_file_path
            
        material_info = export_mesh_subset_to_obj(mesh, subset, material, obj_file_path, mtl_file_path)
                        
        #generate mjcf mesh
        relative_obj_path = os.path.relpath(obj_file_path, self._output_dir_path)
        mjcf_mesh = mjcf.Mesh(
            file=relative_obj_path,
            name=geom_name,
            scale=[float(scale[0]),float(scale[1]),float(scale[2])]
        )
        self.asset.add_child(mjcf_mesh)
        
        #generate mjcf geom
        geometry_visual = mjcf.Geom(
            name=geom_name,
            type="mesh",
            pos=[float(origin[0]),float(origin[1]),float(origin[2])],
            quat=[float(quat_wxyz[0]),float(quat_wxyz[1]),float(quat_wxyz[2]),float(quat_wxyz[3])],
            mesh=geom_name
        )
                        
        if material:
            material_name = material.GetPrim().GetName()  
            if self._generate_mjcf_material(material_info,material_name):
                geometry_visual.material = material_name
                                        

        return geometry_visual
                                 
    def _generate_mjcf_material(self,UsdMaterialWrapper:UsdMaterialWrapper,material_name):
        """Generate mjcf material

        Args:
            UsdMaterialWrapper: usd material wrapper.
            material_name: The name of the material.
        Returns:
            If the material is generated successfully, True or False.
        """
        
        #mjcf material properties        
    
        #rgba Color and transparency of the material. All components should be in the range [0 1].
        #reflectance This attribute should be in the range [0 1].
        #emission mjcf only provide a scalar setting. 
        #specular mjcf only provide a scalar setting.
        #shininess one float value The value given here is multiplied by 128 before passing it to OpenGL, so it should be in the range [0 1].
    
        #diffuse_texture mjcf only support diffuse texture
        
        
        if material_name in self.exported_mjcf_materials:
            return True
        
        mjcf_material_dict = UsdMaterialWrapper.export_mjcf_material()
        
        
        if mjcf_material_dict is not None:
             
            #generate mjcf material
            #'rgb' 'alpha' 'reflectance' 'emission' 'specular' 'shininess'
            
            mjcf_material = None
            if mjcf_material_dict is not None:
                
                mjcf_material = mjcf.Material(
                    name=material_name
                )
                if 'rgba' in mjcf_material_dict and mjcf_material_dict.get('rgba'):
                    mjcf_material.rgba = list(mjcf_material_dict.get('rgba'))
                                        
                # if 'specular' in material_properties and material_properties.get('specular'):
                #     mjcf_material.specular = list(material_properties.get('specular'))
                    
                # if 'shininess' in material_properties and material_properties.get('shininess'):
                #     mjcf_material.shininess = list(material_properties.get('shininess'))
                    
                # if 'reflectance' in material_properties and material_properties.get('reflectance'):
                #     mjcf_material.reflectance = list(material_properties.get('reflectance'))
                    
                # if 'metallic' in material_properties:
                #     mjcf_material.reflectance = material_properties.get('metallic')
                
                # if 'ambient' in material_properties:
                #     ambient = material_properties.get('ambient')
                #     mjcf_material.emission = ambient
                    
                self.asset.add_child(mjcf_material)
                self.exported_mjcf_materials.add(material_name)
                
                #mjcf only supports diffuse texture
                #generate mjcf texture
                if mjcf_material_dict.get('diffuse_texture'):
                    texture_name =mjcf_material_dict.get('diffuse_texture')
                    texture_name = texture_name.replace('.jpg','.png')
                    texture_name = texture_name.replace('.JPG','.png')
                    texture_path = self._mesh_dir_path / 'textures' / texture_name
                    relative_texture_path = os.path.relpath(texture_path, self._output_dir_path)
                    mjcf_texture = mjcf.Texture(
                        name=material_name,
                        file=relative_texture_path,
                        type="2d"
                    )
                    if mjcf_material is not None:
                        mjcf_material.texture = material_name
                    self.asset.add_child(mjcf_texture)          
            
            
            if mjcf_material is not None:
                return True
            else:
                return False
    
    def _add_joint(self, parent_element: mjcf.Element,node: TransformNode) -> None:
        
        if node.prim is None:
            raise RuntimeError("Something is wrong. A joint node shoud always have a prim value.")
        joint_prim = node.prim
        if node.is_root is None:
            raise RuntimeError("Something is wrong. A joint node should never be the root node.")
        
        #generate mjcf joint and set general properties
        mjcf_joint = mjcf.Joint(
            name=node.name
        )
        
        parent_link_t_joint = node.from_edges[0].transform
        joint_t_child_link = node.to_edges[0].transform
        child_link_t_joint = Transform.inverse(joint_t_child_link)
        parent_link_t_child_link = parent_link_t_joint @ joint_t_child_link
        xyz,quat_wxyz = UsdToMjcf._get_mjcf_pose(parent_link_t_child_link)
        joint_axis = prim_helper.get_joint_axis(joint_prim)
        
        if joint_axis is not None:
            child_frame_axis_vector = prim_helper.calculate_joint_axis(joint_axis,child_link_t_joint)
            mjcf_joint.axis = child_frame_axis_vector
        
        friction = prim_helper.get_joint_friction(joint_prim)
        armature = prim_helper.get_joint_armature(joint_prim)
        pos = prim_helper.get_joint_position(joint_prim,1)
         
        if friction:
            mjcf_joint.frictionloss = friction
        if armature:
            mjcf_joint.armature = armature
        mjcf_joint.pos = pos.tolist()
        
        if prim_helper.is_a_fixed_joint(joint_prim) or prim_helper.is_an_unassigned_joint(
            joint_prim
        ):
            
            body = mjcf.Body(
                name=node.name,
                pos=[float(xyz[0]),float(xyz[1]),float(xyz[2])],
                quat=[float(quat_wxyz[0]),float(quat_wxyz[1]),float(quat_wxyz[2]),float(quat_wxyz[3])]
            )
            if parent_element is not None:
                parent_element.add_child(body)          
            return body
        elif prim_helper.is_a_revolute_joint(joint_prim):
            lower_limit, upper_limit = prim_helper.get_joint_limits(joint_prim)
            mjcf_joint.type = "hinge"
            mjcf_joint.limited = True
            mjcf_joint.range = [lower_limit, upper_limit]
                
            body = mjcf.Body(
                name=node.name,
                pos=[float(xyz[0]),float(xyz[1]),float(xyz[2])],
                quat=[float(quat_wxyz[0]),float(quat_wxyz[1]),float(quat_wxyz[2]),float(quat_wxyz[3])]
            )
            if parent_element is not None:
                parent_element.add_child(body)
            body.add_child(mjcf_joint)            
            return body  
            
        elif prim_helper.is_a_prismatic_joint(joint_prim):
            lower_limit, upper_limit = prim_helper.get_joint_limits(joint_prim)
            mjcf_joint.type = "slide"
            mjcf_joint.limited = True
            mjcf_joint.range = [lower_limit, upper_limit]
            
            body = mjcf.Body(
                name=node.name,
                pos=[float(xyz[0]),float(xyz[1]),float(xyz[2])],
                quat=[float(quat_wxyz[0]),float(quat_wxyz[1]),float(quat_wxyz[2]),float(quat_wxyz[3])]
            )
            if parent_element is not None:
                parent_element.add_child(body)
            body.add_child(mjcf_joint)
            return body
        else:
            schema_type = joint_prim.GetPrimTypeInfo().GetSchemaTypeName()
            msg = (
                f"Joint '{node.path}' will be skipped. The '{node.name}' joint is of type"
                f" '{schema_type}', which is not currently supported. Supported joint types:"
                " 'UsdPhysics.FixedJoint', 'UsdPhysics.RevoluteJoint',"
                " 'UsdPhysics.PrismaticJoint'."
            )
            raise RuntimeError(msg)
        
    def _add_body(self,parent_element: mjcf.Element, node: TransformNode):
        xyz, quat_wxyz = UsdToMjcf._get_mjcf_pose(node.world_transform)
        body = mjcf.Body(
            name=node.name,
            pos=[float(xyz[0]),float(xyz[1]),float(xyz[2])],
            quat=[float(quat_wxyz[0]),float(quat_wxyz[1]),float(quat_wxyz[2]),float(quat_wxyz[3])]
        )
        if parent_element is not None:
            parent_element.add_child(body)
        return body
        
    def _add_subtree(self,parent_element: mjcf.Element, node: TransformNode) -> None:
        children = node.from_neighbors
        for child in children:
            if child.type == NodeType.GEOMETRY:
                self._add_geometry(parent_element,child)
            elif (
                child.type == NodeType.LINK
                or child.type == NodeType.SENSOR
                or child.type == NodeType.PHONY
            ):
                current_element =self._add_link(parent_element,child)
                self._add_subtree(parent_element,child)
            elif child.type == NodeType.JOINT:
                current_element =self._add_joint(parent_element,child)
                self._add_subtree(current_element,child)
            
    def _add_link(self,parent_element: mjcf.Element,node: TransformNode):
        if (
            (
                node.type == NodeType.PHONY
                or node.type == NodeType.LINK
                or node.type == NodeType.SENSOR
            )
            and not node.is_root
            and (
                node.to_neighbors[0].type == NodeType.PHONY
                or node.to_neighbors[0].type == NodeType.LINK
            )
        ):
            body =self._add_body(parent_element,node.to_neighbors[0])
            return body

        #link = mjcf.Link(name=node.name)

        if node.type != NodeType.PHONY:
            if node.prim is None:
                raise RuntimeError(
                    "Something is wrong. A link node shoud always have a prim value."
                )
            if "PhysicsMassAPI" in node.prim.GetAppliedSchemas() and node.prim.HasAttribute(
                "physics:mass"
            ):
                mass_val = node.prim.GetAttribute("physics:mass").Get()
                if mass_val > 0:
                    #TODO: Add Inertia to the body
                   print(mass_val)
        return parent_element

    @staticmethod
    def _get_mjcf_pose(transform: np.ndarray)-> Tuple[list[float], list[float]]:
        xyz = list(Transform.get_translation(transform))
        with warnings.catch_warnings():
            msg = (
                "Gimbal lock detected. Setting third angle to zero since it is not possible to"
                " uniquely determine all angles."
            )
            warnings.filterwarnings("ignore", message=msg)
            quat_wxyz = list(Transform.get_rotation(transform, as_quat=True, as_wxyz=True))
        return xyz, quat_wxyz
    
    
    