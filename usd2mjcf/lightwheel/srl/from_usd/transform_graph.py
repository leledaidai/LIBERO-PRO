# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Functions and classes to create a coordinate transform graph."""

# Standard Library
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

# Third Party
import numpy as np
from pxr import Usd

# NVIDIA
import lightwheel.srl.usd.prim_helper as prim_helper
from lightwheel.srl.abc.srl import SRL
from lightwheel.srl.basics.types import Affine, PathLike
from lightwheel.srl.from_usd._from_usd_helper import (
    NodeType,
    _is_geometry_prim,
    _is_joint_prim,
    _is_link_prim,
    _is_sensor_prim,
)
from lightwheel.srl.math.transform import Transform
from lightwheel.srl.tools.misc import optional_import

Digraph = optional_import(
    extras_require_name="graphviz", module_name="graphviz", attr_name="Digraph"
)


class TransformNode:
    """A node in the transform graph."""

    count = 0  # Count how many times the class has been initialized

    def __init__(
        self,
        name: str,
        prim: Optional[Usd.Prim],
        world_transform: Optional[np.ndarray] = None,
    ):
        """Initialize a new `TransformNode` object.

        Args:
            name: Name of the node.
            prim: Prim stored in the node.
        """
        TransformNode.count += 1

        self._id = TransformNode.count
        self._name: str = name
        self._prim: Optional[Usd.Prim] = prim
        self._world_transform = Transform.identity()
        if world_transform is None and prim is not None:
            self._world_transform = prim_helper.get_transform_world(prim)

        if prim is None:
            self._type = NodeType.PHONY
        elif _is_geometry_prim(prim):
            self._type = NodeType.GEOMETRY
        elif _is_joint_prim(prim):
            self._type = NodeType.JOINT
        elif _is_link_prim(prim):
            self._type = NodeType.LINK
        elif _is_sensor_prim(prim):
            self._type = NodeType.SENSOR
        else:
            raise RuntimeError(f"Unknown prim type for '{self.path}'")
        # List of edges that this node is the "from" node for in the graph.
        self._from_edges: List[TransformEdge] = []
        # List of edges that this node is the "to" node for in the graph.
        self._to_edges: List[TransformEdge] = []

    @property
    def world_transform(self):
        return self._world_transform

    @property
    def id(self) -> int:
        """Return this node's ID number."""
        return self._id

    @property
    def name(self) -> str:
        """Return the name associated with this node."""
        return self._name

    @property
    def prim(self) -> Optional[Usd.Prim]:
        """Return the prim associated with this node."""
        return self._prim

    @property
    def path(self) -> Optional[str]:
        """Return the prim path associated with this node."""
        if self._prim is None:
            return None
        return prim_helper.get_path(self._prim)

    @property
    def sku(self) -> str:
        """Return a unique "semi" readable label for the node.

        Used in the graphviz generation.
        """
        return self.path if self.path is not None else str(self.id)

    @property
    def type(self) -> NodeType:
        """Return the node type."""
        return self._type

    @property
    def from_edges(self) -> List["TransformEdge"]:
        """Return the edges that this node is the "from" node for the edges in the graph."""
        return self._from_edges

    @property
    def to_edges(self) -> List["TransformEdge"]:
        """Return the edges that this node is the "to" node for the edges in the graph."""
        return self._to_edges

    @property
    def edges(self) -> List["TransformEdge"]:
        """Return the all edges that this node is connected in the graph."""
        return self._to_edges + self._from_edges

    @property
    def from_neighbors(self) -> List["TransformNode"]:
        """Return a list of all of the "from" nodes this node is directly connected to."""
        return [edge.from_node for edge in self._to_edges]

    @property
    def to_neighbors(self) -> List["TransformNode"]:
        """Return a list of all of the "to" nodes this node is directly connected to."""
        return [edge.to_node for edge in self._from_edges]

    @property
    def neighbors(self) -> List["TransformNode"]:
        """Return a list of all the nodes this node is directly connected to."""
        return self.to_neighbors + self.from_neighbors

    @property
    def is_root(self) -> bool:
        """Return `True` if the node has no "to neighbors."""
        return len(self.to_neighbors) == 0

    @property
    def is_leaf(self) -> bool:
        """Return `True` if the node has no "from neighbors."""
        return len(self.from_neighbors) == 0

    def __repr__(self) -> str:
        """Return the string representation of this node."""
        return f"TransformNode({self.name})"


class TransformEdge:
    """An edge in the TransformGraph.

    The transform stored here is the transform of coordinates from the  `from_node` frame to the
    `to_node` frame (i.e.  `to_node<-from_node`). Thus, it is the transform of the reference frame
    for the `from_node` with respect to the reference frame of the `to_node`.
    """

    count = 0  # Count how many times the class has been initialized

    def __init__(
        self,
        to_node: TransformNode,
        from_node: TransformNode,
        transform: Optional[np.ndarray] = None,
        # from_world_transform: Optional[np.ndarray] = None,
        # to_world_transform: Optional[np.ndarray] = None
    ):
        """Initialize a new `TransformNode` object.

        Args:
            to_node: The "to node" in the edge of the graph.
            from_node: The "from node" in the edge of the graph.
            transform: The transform from the `from_node` to the `to_node` (i.e.
                to_node<-from_node`). Defaults to identity.
        """
        TransformEdge.count += 1

        self._id = TransformEdge.count
        self._to_node = to_node
        self._from_node = from_node
        self._transform = transform if transform is not None else Transform.identity()
        self._from_node._from_edges.append(self)
        self._to_node._to_edges.append(self)
    #     self._from_world_transform = from_world_transform
    #     self._to_world_transform = to_world_transform

    # @property
    # def from_world_transform(self) -> Optional[np.ndarray]:
    #     return self._from_world_transform

    # @property
    # def to_world_transform(self) -> Optional[np.ndarray]:
    #     return self._to_world_transform

    @property
    def id(self) -> int:
        """Return this edge's ID number."""
        return self._id

    @property
    def to_node(self) -> TransformNode:
        """Return the to_node associated with this edge."""
        return self._to_node

    @property
    def from_node(self) -> TransformNode:
        """Return the from_node associated with this edge."""
        return self._from_node

    @property
    def transform(self) -> np.ndarray:
        """Return the transform associated with this edge.

        Transform is `to_node<-from_node`.
        """
        return self._transform

    @property
    def name(self) -> str:
        """Return the name associated with this edge.

        The name is always in the form "{to_node.name}___{from_node.name}".
        """
        return f"{self.to_node.name}<-{self.from_node.name}"

    @property
    def sku(self) -> str:
        """Return the SKU associated with this edge.

        The SKU is always in the form "{to_node.sku}___{from_node.sku}".
        """
        return f"{self.to_node.sku}<-{self.from_node.sku}"

    def __repr__(self) -> str:
        """Return the string representation of this edge."""
        return f"TransformEdge({self.name})"


class TransformGraph(SRL):
    """A double linked graph to represent the transforms between prims for the given stage."""

    PRIM_TYPES_EXCLUSION = [
        "distantlight",
        "geomsubset",
        "material",
        "physicsscene",
        "scope",
        "shader",
    ]

    def __init__(
        self,
        name: str,
        **kwargs: Any,
    ):
        """Initialize a new `TransformGraph` object.

        Args:
            name: Name of the graph.
            kwargs: Additional keyword arguments are passed to the parent class,
                :class:`~nvidia.srl.abc.srl.SRL`.
        """
        # Initialize parent class
        super().__init__(**kwargs)
        self.logger.debug(f"Creating TransformGraph: '{name}'.")

        # Initialize instance variables
        self.name = name

        # Initialize instance variables that are used when initialized from stage
        self._stage: Optional[Usd.Stage] = None
        self._parent_link_is_body_1: List[str] = []
        self._used_parent_link_is_body_1: List[str] = []
        self._duplicate_names: Set[str] = set()

        # A record of all nodes in the graph. The key is the node SKU, and the value is the node.
        self._node_dict: Dict[str, TransformNode] = {}

        # A record of all the edges in the graph. The key is the edge SKU, and the value is the
        # edge.
        self._edge_dict: Dict[str, TransformEdge] = {}

        # Flag to keep track if the graph nodes have been sorted
        self._sorted = False

    @classmethod
    def init_from_stage(
        cls,
        stage: Usd.Stage,
        name: Optional[str] = None,
        parent_link_is_body_1: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "TransformGraph":
        """Create new `TransformGraph` object for the given stage.

        Args:
            stage: USD stage to build transform graph for.
            name: Name of the graph. Defaults to the USD stage root layer file base name.
            parent_link_is_body_1: A list of joint node names where the parent link is assumed to be
                the body 1 target prim, instead of the default body 0 target prim. Note, when only
                one body target is set, then the parent link is assumed to be the default prim, and
                the child link is the prim of whatever body target that is set.
            kwargs: Additional keyword arguments are passed to the parent class,
                :class:`~nvidia.srl.abc.srl.SRL`.

        Returns:
            TransformGraph: New `TransformGraph` object initialized from a USD stage.
        """
        if name is None:
            name = Path(stage.GetRootLayer().resolvedPath.GetPathString()).stem

        # Initialize class
        transform_graph = cls(name=name, **kwargs)

        # Initialize instance variables
        transform_graph._stage = stage
        if parent_link_is_body_1 is not None:
            transform_graph._parent_link_is_body_1 = parent_link_is_body_1

        # Build transform graph
        transform_graph._build_transform_graph()

        # Check if any parent_link_body_1 items were not used
        not_used_parent_link_body_1 = set(transform_graph._parent_link_is_body_1) - set(
            transform_graph._used_parent_link_is_body_1
        )
        if not_used_parent_link_body_1:
            msg = (
                "Some joint nodes were not found in the `parent_link_is_body_1` list that was"
                f" provided. Joints not found:\n{not_used_parent_link_body_1}"
            )
            transform_graph.logger.warning(msg)

        return transform_graph

    @classmethod
    def init_from_usd_path(cls, usd_path: PathLike, **kwargs: Any) -> "TransformGraph":
        """Create new `TransformGraph` object for the stage at the given USD path.

        Args:
            usd_path: File path to load the USD from.

        Returns:
            TransformGraph: New `TransformGraph` object initialized from a USD path.
        """
        stage = prim_helper.open_stage(usd_path)
        return cls.init_from_stage(stage, **kwargs)

    @property
    def nodes(self) -> List[TransformNode]:
        """Return a list of all the nodes in the transform graph."""
        return list(self._node_dict.values())

    @property
    def node(self) -> Dict[str, TransformNode]:
        """Return a dict of the nodes in the transform graph.

        The keys for the dict are the node IDs.
        """
        return self._node_dict

    @property
    def edges(self) -> List[TransformEdge]:
        """Return a list of all the nodes in the transform graph."""
        return list(self._edge_dict.values())

    @property
    def edge(self) -> Dict[str, TransformEdge]:
        """Return a dict of the edges in the transform graph.

        The keys for the dict are the node names.
        """
        return self._edge_dict

    @property
    def sorted_nodes(self) -> List[TransformNode]:
        """Return a list of all the nodes in the transform tree sorted in a depth-first order."""
        if not self._sorted:
            self._build_depth_first_list()
        return self._depth_first_list

    def get_node(self, name_or_path: str) -> TransformNode:
        """Return the node with the given name or path.

        Note:
            If the node name is provided then this loops through all nodes looking for the node with
            the given name.
        """
        if name_or_path[0] == "/":
            path = name_or_path
            node = self.get_node_from_path(path)
        else:
            name = name_or_path
            node = self.get_node_from_name(name)

        return node

    def get_node_from_name(self, name: str) -> TransformNode:
        """Return the node with the given node name.

        Note:
            This loops through all nodes looking for the node with the given name.
        """
        nodes_with_name = [node for node in self.nodes if node.name == name]
        assert len(nodes_with_name) <= 1  # Node names should be unique
        if len(nodes_with_name) == 0:
            msg = (
                f"A node with the name '{name}' does not exist in the graph. Valid"
                f" node names:\n{self.list_node_names()}"
            )
            raise ValueError(msg)
        return nodes_with_name[0]

    def get_node_from_path(self, path: str) -> TransformNode:
        """Return the node with the given prim path.

        Note:
            This loops through all nodes looking for the node with the given prim path.
        """
        nodes_with_path = [node for node in self.nodes if node.path == path]
        assert len(nodes_with_path) <= 1  # Node names should be unique
        if len(nodes_with_path) == 0:
            msg = (
                f"A node with the path '{path}' does not exist in the graph. Valid"
                f" node path:\n{self.list_node_paths()}"
            )
            raise ValueError(msg)
        return nodes_with_path[0]

    def get_edge(self, name_or_sku: str) -> TransformEdge:
        """Return the edge with the given name or SKU.

        Note:
            If the edge name is provided then this loops through all edges looking for the edge with
            the given name.
        """
        if name_or_sku[0] == "/":
            sku = name_or_sku
            edge = self.edge[sku]
        else:
            name = name_or_sku
            edge = self.get_edge_from_name(name)

        return edge

    def get_edge_from_name(self, name: str) -> TransformEdge:
        """Return the edge with the given edge name.

        Note:
            This loops through all edges looking for the edge with the given name.
        """
        edges_with_name = [edge for edge in self.edges if edge.name == name]
        assert len(edges_with_name) <= 1  # Edge names should be unique
        if len(edges_with_name) == 0:
            msg = (
                f"An edge with the name '{name}' does not exist in the graph. Valid"
                f" edge names:\n{self.list_edge_names()}"
            )
            raise ValueError(msg)
        return edges_with_name[0]

    def get_roots(self) -> List[TransformNode]:
        """Return all root nodes in the graph."""
        root_nodes = list(filter(lambda node_: node_.is_root, self.nodes))
        return root_nodes

    def get_leafs(self) -> List[TransformNode]:
        """Return all leaf nodes in the graph."""
        root_nodes = list(filter(lambda node_: node_.is_leaf, self.nodes))
        return root_nodes

    def is_possible_tree(self) -> bool:
        """Return `True` if the graph is has tree structure (i.e. acyclic directed graph).

        Note:
            Doesn't check if the graph is fully connected.
        """
        # Check that there is only one root
        single_root = len(self.get_roots()) == 1
        if not single_root:
            return False

        # Check that each node (besides the root) only has one "to neighbor"
        non_root_nodes = [node_ for node_ in self.nodes if not node_.is_root]
        only_one_to_neighbor = bool(
            np.all(np.array([len(node_.to_neighbors) == 1 for node_ in non_root_nodes]))
        )

        return only_one_to_neighbor

    def is_possible_urdf(self) -> bool:
        """Return `True` if the graph can be converted to a URDF.

        Needs to be a tree and all joint nodes need one be connect to one link for the "to node" and
        one link for the "from node".
        """
        # Check if it is a tree
        if not self.is_possible_tree():
            return False

        # Check joint connections
        joint_nodes = [node_ for node_ in self.nodes if node_.type == NodeType.JOINT]
        all_valid_joints = bool(
            np.all(
                np.array(
                    [
                        len(node_.to_neighbors) == 1 and len(node_.from_neighbors)
                        for node_ in joint_nodes
                    ]
                )
            )
        )

        return all_valid_joints
    
    def is_possible_mjcf(self) -> bool:
        """Return `True` if the graph can be converted to a MJCF.

        Needs to be a tree and all joint nodes need one be connect to one link for the "to node" and
        one link for the "from node".
        """
        # Check if it is a tree
        if not self.is_possible_tree():
            return False

        # Check joint connections
        joint_nodes = [node_ for node_ in self.nodes if node_.type == NodeType.JOINT]
        all_valid_joints = bool(
            np.all(
                np.array(
                    [
                        len(node_.to_neighbors) == 1 and len(node_.from_neighbors)
                        for node_ in joint_nodes
                    ]
                )
            )
        )

        return all_valid_joints

    def list_node_names(self) -> List[str]:
        """List all the node names in the graph."""
        return [node.name for node in self.nodes]

    def list_node_paths(self) -> List[str]:
        """List all the node paths in the graph."""
        return [node.sku for node in self.nodes]

    def list_edge_names(self) -> List[str]:
        """List all the edge names in the graph."""
        return [edge.name for edge in self.edges]

    def list_edge_paths(self) -> List[str]:
        """List all the edge paths in the graph."""
        return [edge.sku for edge in self.edges]

    def add_node(self, node: TransformNode) -> None:
        """Add the given node to the graph."""
        if node.path in self.list_node_paths():
            msg = f"A node with the path '{node.path}' already existed in the graph."
            self.logger.warning(msg)
        self._node_dict[node.sku] = node
        self._sorted = False

    def add_edge(self, edge: TransformEdge) -> None:
        """Add an edge to the graph.

        If either of the nodes in the edge are not already in the graph, the nodes will be added to
        the graph.

        Args:
            edge: The edge to add to the graph.
        """
        if edge.to_node not in self.nodes:
            self.add_node(edge.to_node)
        if edge.from_node not in self.nodes:
            self.add_node(edge.from_node)
        if edge.sku in self.list_edge_paths():
            msg = f"An edge with the path '{edge.sku}' already exists in the graph."
            self.logger.warning(msg)
        self._edge_dict[edge.sku] = edge
        self._sorted = False

    def remove_edge(self, edge: TransformEdge) -> None:
        """Remove the given edge from the graph."""
        if edge.sku not in self._edge_dict.keys():
            msg = f"The edge with ID {edge.sku} does not exist in the graph."
            self.logger.warning(msg)
            return
        edge.to_node.to_edges.remove(edge)
        edge.from_node.from_edges.remove(edge)
        del self._edge_dict[edge.sku]

        self.logger.debug(f"Removing edge '{edge.name}'.")
        self._sorted = False

    def remove_node(self, node: TransformNode) -> None:
        """Remove the given node from the graph."""
        if node.path not in self.list_node_paths():
            msg = f"The node with the path '{node.path}' does not exist in the graph."
            self.logger.warning(msg)
        for edge in node.edges:
            self.remove_edge(edge)
        del self._node_dict[node.sku]

        self.logger.debug(f"Removing node '{node.name}'.")
        self._sorted = False

    @staticmethod
    def _get_transform_relative(to_node: TransformNode, from_node: TransformNode) -> Affine:
        """Get the relative pose between the two node frames."""
        return prim_helper.get_transform_relative(from_node.prim, to_node.prim)

    def connect_nodes(
        self,
        to_node: TransformNode,
        from_node: TransformNode,
        transform: Optional[np.ndarray] = None,
    ) -> None:
        """Connect two nodes in the graph by creating an edge between them.

        If either of the nodes in the edge are not already in the graph, the nodes will be added to
        the graph.

        Args:
            to_node: The "to" node for the edge.
            from_node: The "from" node for the edge.
            transform: The transform for the edge. Defaults to relative Xform transform
                between the respective node prims.
        """
        if transform is None:
            transform = self._get_transform_relative(to_node, from_node)

        from_world_transform = from_node.world_transform
        to_world_transform = to_node.world_transform
        edge = TransformEdge(to_node=to_node, from_node=from_node, transform=prim_helper.get_transform_relative_by_trans(from_world_transform, to_world_transform))
        self.add_edge(edge)

    def __str__(self) -> str:
        return f"TransformGraph({self.name})"

    def __repr__(self) -> str:
        repr_list = [f"TransformGraph({self.name})"]
        repr_list.append("Nodes:")
        repr_list.extend(self.list_node_names())
        repr_list.append("")
        repr_list.append("Edges:")
        repr_list.extend(self.list_edge_names())
        return "\n".join(repr_list)

    def _build_transform_graph(self) -> None:
        if self._stage is None:
            msg = "The stage has not been set."
            raise RuntimeError(msg)

        # Make all prims not instanceable before proceeding
        # This allows for the prim children to be detected
        for prim in self._stage.Traverse():
            if prim.IsInstanceable():
                prim.SetInstanceable(False)

        # Record duplicate prim names
        prim_names = [
            _prim.GetName()
            for _prim in prim_helper.get_prims(
                stage=self._stage, prim_types_exclusion=self.PRIM_TYPES_EXCLUSION
            )
            if isinstance(_prim, Usd.Prim)
        ]
        seen = set()
        dupes = set([x for x in prim_names if x in seen or seen.add(x)])  # type: ignore
        self._duplicate_names = dupes

        # Create all nodes
        for prim in prim_helper.get_prims(
            stage=self._stage, prim_types_exclusion=self.PRIM_TYPES_EXCLUSION
        ):
            try:
                self._create_node(prim)
            except RuntimeError as err:
                self.logger.info(str(err))

        # Update joint nodes
        for prim in TransformGraph._get_joints_for_stage(self._stage):
            self._update_joint_node(prim)

        # Update all other nodes
        for prim in prim_helper.get_prims(
            stage=self._stage, prim_types_exclusion=self.PRIM_TYPES_EXCLUSION
        ):
            self._update_link_node(prim)

        return

    def _create_node(self, prim: Usd.Prim) -> TransformNode:
        if not TransformGraph._is_link_type(prim) and not prim_helper.is_a_joint(prim):
            msg = (
                f"Prim '{prim.GetPath().pathString}' (type: '{prim.GetTypeName()}') not added to"
                " `TransformGraph` because it is not a link type or joint type prim."
            )
            raise RuntimeError(msg)

        # Check if entity is already in the graph, if so then return
        path = prim_helper.get_path(prim)
        try:
            return self.get_node_from_path(path)
        except ValueError:
            pass

        # Make unique node name
        name = prim.GetName()
        old_name = name
        a_prim = prim
        while a_prim.GetName() in self._duplicate_names or ("Joint" in name and name == old_name):
            if a_prim.GetParent().GetPath().pathString == "/":
                break
            a_prim = a_prim.GetParent()
            name = f"{a_prim.GetName()}_{name}"
        # Create node and add it
        node = TransformNode(name=name, prim=prim)
        self.add_node(node)

        return node

    def _update_joint_node(self, joint_prim: Usd.Prim) -> None:
        if self._stage is None:
            msg = "The stage has not been set."
            raise RuntimeError(msg)

        if not prim_helper.is_a_joint(joint_prim):
            return

        # Get joint node from graph
        joint_path = prim_helper.get_path(joint_prim)
        joint_node = self.get_node_from_path(joint_path)

        parent_link_is_body_1_flag = joint_node.name in self._parent_link_is_body_1
        if parent_link_is_body_1_flag:
            self._used_parent_link_is_body_1.append(joint_node.name)

        # Get joint links
        body_0_link_prim, body_1_link_prim = prim_helper.get_links_for_joint(joint_prim)

        if body_0_link_prim is None and body_1_link_prim is None:
            self.logger.warning(f"Joint prim at '{joint_path}' does not have any body targets set.")
            return
        elif body_0_link_prim is not None and body_1_link_prim is None:
            parent_link_prim = self._stage.GetDefaultPrim()
            child_link_prim = body_0_link_prim
        elif body_0_link_prim is None and body_1_link_prim is not None:
            parent_link_prim = self._stage.GetDefaultPrim()
            child_link_prim = body_1_link_prim
        else:
            if parent_link_is_body_1_flag:
                child_link_prim = body_0_link_prim
                parent_link_prim = body_1_link_prim
            else:
                parent_link_prim = body_0_link_prim
                child_link_prim = body_1_link_prim

        # Update parent link
        if parent_link_prim :
            parent_link___joint = prim_helper.get_joint_transform(joint_prim, 0)
            parent_link_path = prim_helper.get_path(parent_link_prim)
            parent_link_node = self.get_node_from_path(parent_link_path)
            self.connect_nodes(
                to_node=parent_link_node, from_node=joint_node, transform=parent_link___joint
            )

        # Update child link
        if child_link_prim is not None and child_link_prim.IsValid():
            child_link___joint = prim_helper.get_joint_transform(joint_prim, 1)
            joint___child_link = Transform.inverse(child_link___joint)
            child_link_path = prim_helper.get_path(child_link_prim)
            child_link_node = self.get_node_from_path(child_link_path)
            self.connect_nodes(
                to_node=joint_node, from_node=child_link_node, transform=joint___child_link, 
            )

        return

    def _update_link_node(self, link_prim: Usd.Prim) -> None:
        if not TransformGraph._is_link_type(link_prim):
            return

        # Get entity node from graph
        link_path = prim_helper.get_path(link_prim)
        link_node = self.get_node_from_path(link_path)

        parent_prim = link_prim.GetParent()
        if prim_helper.is_root(parent_prim):
            return

        parent_path = prim_helper.get_path(parent_prim)
        try:
            parent_node = self.get_node_from_path(parent_path)
        except ValueError:
            parent_node = self._create_node(parent_prim)
        self.connect_nodes(parent_node, link_node)

        return

    def _build_depth_first_list(self) -> None:
        self._depth_first_list: List[TransformNode] = []
        if not self.is_possible_tree():
            msg = "Cannot build a depth first list for graphs that are not trees."
            raise RuntimeError(msg)

        def _loop_depth_first(root_node: TransformNode) -> None:
            self._depth_first_list.append(root_node)
            for node in root_node.from_neighbors:
                _loop_depth_first(node)

        _loop_depth_first(self.get_roots()[0])
        self._sorted = True

    @staticmethod
    def get_subtree_nodes(root_node: TransformNode) -> List[TransformNode]:
        """Return a list of nodes that make the subtree for the given root node of the subtree."""
        subtree_nodes = []

        def _loop_depth_first(root_: TransformNode) -> None:
            if len(root_.to_neighbors) > 1:
                to_neighbor_names = list(map(lambda node_: node_.name, root_.to_neighbors))
                msg0 = (
                    f"Cannot build a subtree for the root node '{root_node.name}' because node"
                    f"\n'{root_.name}' has more than one 'to-neighbors':\n{to_neighbor_names}."
                )
                msg1 = (
                    "What this means is that the kinematics of the USD cannot be coerced into a"
                    "\ntree structure. Usually this happens when a link is connected to more than"
                    "\none joint. To resolve this issue remove nodes and/or edges in the transform"
                    "\ngraph, thus breaking the kinematic loops. For example, removing all but one"
                    "\nof the notes listed above will resolve the issue."
                )
                msg2 = (
                    "Reviewing the Graphviz image of the full transform graph is helpful in"
                    "\ndeciding what nodes and/or edges to remove. The Graphviz image of the full"
                    "\ntransform graph can be generated with the `usd_to_graphviz` command"
                )
                msg = "\n".join([msg0, msg1, msg2])
                raise RuntimeError(msg)

            subtree_nodes.append(root_)
            for node in root_.from_neighbors:
                _loop_depth_first(node)

        _loop_depth_first(root_node)
        return subtree_nodes

    @staticmethod
    def _get_joints_for_stage(
        stage: Usd.Stage, joint_selector_func: Optional[Callable] = None
    ) -> List[Usd.Prim]:
        """Get all the child joint prims from the given stage.

        This returns a list of all joints associated with the given articulated object. See
        :func:`~srl.util.prim_helper.get_child_joints_for_link` to get immediate child joint prims
        for a given link prim.

        Args:
            stage: USD stage to get the joints for
            joint_selector_func: Filter function to select which joints from the articulated root to
            return.
        """
        # NOTE (roflaherty): This is should share code from
        # prim_helper.get_joints_for_articulated_root
        if joint_selector_func is None:
            joint_selector_func = prim_helper.is_a_joint
        joint_prims = []
        for joint in filter(joint_selector_func, stage.Traverse()):
            if any(link is not None for link in prim_helper.get_links_for_joint(joint)):
                joint_prims.append(joint)
        return joint_prims

    @staticmethod
    def _is_link_type(prim: Usd.Prim) -> bool:
        # A prim without a type is assumed to be of type Xform
        if prim.GetTypeName() == "":
            return True

        valid_type_names = [
            "Xform",
            "Cylinder",
            "Cube",
            "Sphere",
            "Mesh",
            "Camera",
            "IsaacImuSensor",
        ]

        return prim.GetTypeName() in valid_type_names

    def build_graphviz(self) -> Digraph:  # type: ignore
        """Build a return graphviz Digraph object.

        Return:
            The Transform graph as graphviz object.
        """
        dot = Digraph(comment=self.name)
        dot.attr(label=self.name)

        # Create graph nodes
        for node in self.nodes:
            if node.type == NodeType.GEOMETRY:
                dot.attr("node", shape="hexagon")
            elif node.type == NodeType.JOINT:
                dot.attr("node", shape="oval")
            elif node.type == NodeType.LINK:
                dot.attr("node", shape="box")
            elif node.type == NodeType.SENSOR:
                dot.attr("node", shape="diamond")
            elif node.type == NodeType.PHONY:
                dot.attr("node", shape="house")
            else:
                raise RuntimeError(f"Unknown node type for '{node.path}'")
            dot.node(str(node.sku), node.name)

        # Create graph edges
        for edge in self.edges:
            dot.edge(str(edge.from_node.sku), str(edge.to_node.sku))

        return dot

    def save_graphviz(
        self, output_dir: PathLike, name: Optional[str] = None, quiet: bool = False
    ) -> Path:
        """Save transform graph as graphviz dot file.

        The transform graph name is used for the names of the output files.

        Note: `graphviz` is required to be installed.

        Args:
            output_dir: Directory to where output files will be saved.
            name: Dot file name (without the extension `.dot`). Defaults to TransformGraph
                name.
            quiet: If true, do not print info to terminal.

        Returns:
            Full path to where output file is saved.
        """
        output_dir = Path(output_dir)

        # Build graphviz Digraph object
        dot = self.build_graphviz()

        # Save dot file
        if name is None:
            name = self.name
        filename = name + ".dot"
        filepath = output_dir / filename
        dot.save(filepath)  # type: ignore

        # Print info
        if not quiet:
            msg = f"Transform graph graphviz object saved to:\n{filepath}"
            self.logger.info(msg)

        return filepath

    def render_graphviz(
        self,
        output_dir: PathLike,
        name: Optional[str] = None,
        format: str = "png",
        quiet: bool = False,
    ) -> Path:
        """Render transform graph as graphviz file and as a PNG image.

        The transform graph name is used for the names of the output files.

        Note: `graphviz` is required to be installed.

        Args:
            output_dir: Directory to where output files will be saved.
            name: Rendered image file name (without the extension). Defaults to TransformGraph
                name.
            format: Output format. See https://graphviz.org/docs/outputs/
            quiet: If true, do not print info to terminal.
        """
        output_dir = Path(output_dir)

        # Build graphviz Digraph object
        dot = self.build_graphviz()

        # Render image
        dot.format = format  # type: ignore
        if name is None:
            name = self.name
        # fmt: off
        dot.render(  # type: ignore
            filename=name,
            directory=str(output_dir),
            view=False,
            cleanup=True
        )
        # fmt: on

        # Print info
        if not quiet:
            gv_output_path = output_dir / name
            image_output_path = f"{gv_output_path}.{format}"
            msg = f"Transform graph graphviz object rendered to:\n{image_output_path}"
            self.logger.info(msg)

        return Path(image_output_path)
