# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Helper functions for the Transform Graph class."""

# Standard Library
import copy
from typing import List, Optional

# Third Party
import numpy as np

# NVIDIA
from lightwheel.srl.from_usd._from_usd_helper import NodeType, _get_prim_scale
from lightwheel.srl.from_usd.transform_graph import TransformEdge, TransformGraph, TransformNode
from lightwheel.srl.math.transform import Transform

np.set_printoptions(suppress=True)


def reduce_to_mjcf(
    graph: TransformGraph,
    nodes_to_remove: Optional[List[TransformNode]] = None,
    edges_to_remove: Optional[List[TransformEdge]] = None,
    root_node: Optional[TransformNode] = None,
    do_trim_joint_skipping_edges: bool = True,
    do_trim_to_largest_subtree: bool = True,
    do_check_and_fix_geometry_nodes: bool = True,
    do_squash_consecutive_links: bool = True,
    do_align_joint_frames_with_child_frames: bool = True,
) -> None:
    """Reduce the graph to be MJCF compliant and optimal.

    This involves attempting make the graph into a single tree by removing loop edges and smaller
    trees, and also remove nodes that create additional fixed links in the MJCF file.

    Args:
        graph: The graph object to reduce.
        nodes_to_remove: List of nodes to remove from the `TransformGraph` to break kinematic loops
            and make the graph transformable to something valid to create a MJCF.
        edges_to_remove: List of edges to remove from the `TransformGraph` to break kinematic loops
            and make the graph transformable to something valid to create a MJCF.
        root_node: The root node that will be set as the root of the kinematic structure of the new
            MJCF.  This sets the "robot" element in the new MJCF. The root node can either be
            specified with the prim path or with the node name.
        do_trim_joint_skipping_edges: If true, the `trim_joint_skipping_edges` function is called.
        do_trim_to_largest_subtree: If true, the `trim_to_largest_subtree` function is called.
        do_check_and_fix_geometry_nodes: If true, the `check_and_fix_geometry_nodes` function is
            called.
        do_squash_consecutive_links: If true, the `squash_consecutive_links` function is called.
        do_align_joint_frames_with_child_frames: If true, the `align_joint_frames_with_child_frames`
            function is called.
    """
    # Remove edges
    if edges_to_remove is not None:
        for edge in edges_to_remove:
            graph.remove_edge(edge)

    # Remove nodes
    if nodes_to_remove is not None:
        for node in nodes_to_remove:
            graph.remove_node(node)

    # Remove edges that connect a link or a geometry node to another link or geometry node if at
    # least one of the edges from the node connect to a joint
    if do_trim_joint_skipping_edges:
        trim_joint_skipping_edges(graph)

    if root_node is None:
        if do_trim_to_largest_subtree:
            # Keep only the nodes that are part of the largest subtree
            trim_to_largest_subtree(graph)
    else:
        # Keep only the nodes that are part of the subtree
        trim_to_subtree_root(graph, root_node)

    # Fix geometry nodes that are not leafs or not connected to links
    if do_check_and_fix_geometry_nodes:
        check_and_fix_geometry_nodes(graph)

    # Squash consecutive links
    if do_squash_consecutive_links:
        squash_consecutive_links(graph)

    # Align joint coordinate frames with child link coordinates frames
    if do_align_joint_frames_with_child_frames:
        align_joint_frames_with_child_frames(graph)

    return


def trim_joint_skipping_edges(graph: TransformGraph) -> None:
    """Conditionally trim edges that connect a link or geometry to a link or geometry.

    Edges only get trimmed if the "from_node" of the edge is connected to a joint.

    Loop through all link and geometry nodes, if any of the "to-neighbors" of the node are joint
    nodes, then remove all the edges to the non-joint nodes.

    Args:
        graph: The transform graph to be trimmed.
    """
    nodes = filter(
        lambda node_: node_.type == NodeType.LINK or node_.type == NodeType.GEOMETRY, graph.nodes
    )
    for node in nodes:
        for to_node in node.to_neighbors:
            # Check if this node has any joint nodes as "to neighbors"
            if to_node.type == NodeType.JOINT:
                break
        else:
            # There no joint nodes as "to neighbors", continue to next link node
            continue

        # Remove all edges that not connected to the joint node as "to neighbors"
        for edge in node.from_edges:
            if edge.to_node.type != NodeType.JOINT:
                graph.remove_edge(edge)


def trim_to_largest_subtree(graph: TransformGraph) -> None:
    """Trim to the largest subtree of the graph.

    Note:
        The graph should be partitioned into separate trees, and only trees.

    Args:
        graph: The transform graph to be trimmed.
    """
    largest_subtree_node_cnt = 0
    roots = graph.get_roots()
    if len(roots) == 0:
        msg0 = (
            "Unable to trim to the largest subtree because the transform graph is not partitioned"
            "\ninto separate trees. This usually means there are loops in the graph. Remove the"
            "\nloops by removing nodes and/or edges in the graph."
        )
        msg1 = (
            "Reviewing the Graphviz image of the full transform graph is helpful in deciding what"
            "\nnodes and/or edges to remove. The Graphviz image of the full transform graph can be"
            "\ngenerated with the `usd_to_graphviz` command."
        )
        msg = "\n".join([msg0, msg1])

        raise RuntimeError(msg)
    for root_node in graph.get_roots():
        subtree_nodes = graph.get_subtree_nodes(root_node)
        if len(subtree_nodes) > largest_subtree_node_cnt:
            largest_subtree_node_cnt = len(subtree_nodes)
            largest_subtree_root_node = root_node

    trim_to_subtree_root(graph, largest_subtree_root_node)


def trim_to_subtree_root(graph: TransformGraph, root_node: TransformNode) -> None:
    """Trim the graph to a tree with the root starting at the given node."""
    subtree_nodes = graph.get_subtree_nodes(root_node)
    for node in graph.nodes:
        if node not in subtree_nodes:
            graph.remove_node(node)


def squash_consecutive_links(graph: TransformGraph) -> None:
    """Combine consecutive links together.

    The new link name will be the name of the "to neighbor" link. The transforms and all edges are
    updated correctly.
    """
    # `this_node` is the current link node that is being checked to see if it has any link nodes as
    # "from neighbors". If it has a link node as a "from neighbor" then that node is saved in the
    # `from_node` variable. The edge that connects `this_node` (the "to node") to `from_node` (the
    # "from node") is the `this_node_to_edge`. Each of `from_node`'s "from neighbors" are then
    # connected to `this_node` and the `from_node` is removed. The current "from neighbor" of
    # `from_node` is stored in the `from_from_node` variable. The edge that connects `from_node`
    # (the "to node") to `from_from_node` (the "from node") is the `from_node_to_edge`.

    keep_squashing = True

    while keep_squashing:
        keep_squashing = False
        node_removed = False

        # Loop through all nodes starting from the leaves
        for this_node in reversed(graph.sorted_nodes):
            # Skip all non-link nodes
            if this_node.type != NodeType.LINK and this_node.type != NodeType.PHONY:
                continue

            # Loop through this node's "to edges"
            for this_node_to_edge in this_node.to_edges:
                from_node = this_node_to_edge.from_node
                if from_node.type == NodeType.LINK or from_node.type == NodeType.PHONY:
                    # Loop through this node's from node's "to edges"
                    for from_node_to_edge in from_node.to_edges:
                        from_from_node = from_node_to_edge.from_node

                        this_node___from_node = this_node_to_edge.transform
                        from_node___from_from_node = from_node_to_edge.transform
                        this_node___from_from_node = (
                            this_node___from_node @ from_node___from_from_node
                        )

                        graph.connect_nodes(
                            to_node=this_node,
                            from_node=from_from_node,
                            transform=this_node___from_from_node,
                        )

                    graph.remove_node(from_node)

                    node_removed = True
                    keep_squashing = True

                    break

            if node_removed:
                break


def check_and_fix_geometry_nodes(graph: TransformGraph) -> None:
    """Check that the geometry nodes are leaf nodes and not directly connected to joint nodes.

    If they are fix them by creating phony link nodes as necessary.
    """
    graph_nodes = graph.nodes
    for geom_node in filter(lambda node_: node_.type == NodeType.GEOMETRY, graph_nodes):
        scale = _get_prim_scale(geom_node.prim)
        if len(geom_node.to_neighbors) != 1:
            msg = (
                "The geometry node '{geom_node.path}' does not have exactly one to-neighbor. This"
                " should never happen."
            )
            raise RuntimeError(msg)

        if not geom_node.is_leaf:
            msg = f"'{geom_node.path}' is not a leaf node. Adding phony link node."
            graph.logger.debug(msg)

            # Add phony link node
            name = geom_node.name
            geom_node_link = TransformNode(name, None, geom_node.world_transform)

            to_edges = copy.copy(geom_node.to_edges)
            for to_edge in to_edges:
                rot = Transform.get_rotation(to_edge.transform)
                trans = Transform.get_translation(to_edge.transform) * scale
                transform = Transform.from_rotmat(rot, trans)
                graph.remove_edge(to_edge)
                graph.connect_nodes(geom_node_link, to_edge.from_node, transform)

            from_edges = copy.copy(geom_node.from_edges)
            for from_edge in from_edges:
                # TODO (roflaherty): Same here... see comment above.
                rot = Transform.get_rotation(to_edge.transform)
                trans = Transform.get_translation(to_edge.transform) * scale
                transform = Transform.from_rotmat(rot, trans)
                graph.remove_edge(from_edge)
                graph.connect_nodes(from_edge.to_node, geom_node_link, transform)

            graph.connect_nodes(geom_node_link, geom_node, Transform.identity())

        geom_to_neighbor = geom_node.to_neighbors[0]
        if geom_to_neighbor.type != NodeType.LINK and geom_to_neighbor.type != NodeType.PHONY:
            msg = (
                f"The to-neighbor node of '{geom_node.path}', '{geom_to_neighbor.path}', is not a"
                " link node. Adding phony link node."
            )
            graph.logger.debug(msg)

            # Add phony link node
            name = geom_node.name
            geom_node_link = TransformNode(name, None, geom_node.world_transform)
            geom_node_edge = geom_node.from_edges[0]
            transform = geom_node_edge.transform
            graph.remove_edge(geom_node_edge)
            graph.connect_nodes(geom_node_link, geom_node, Transform.identity())
            graph.connect_nodes(geom_to_neighbor, geom_node_link, transform)


def align_joint_frames_with_child_frames(graph: TransformGraph) -> None:
    """Check that the joint child transforms are identity, if not then update them."""
    for joint_node in filter(lambda node_: node_.type == NodeType.JOINT, graph.nodes):
        for idx, to_edge in enumerate(joint_node.to_edges):
            if not np.allclose(to_edge._transform, Transform.identity()):
                from_node = to_edge.from_node
                for from_node_to_edge in from_node.to_edges:
                    from_node_to_edge._transform = to_edge._transform @ from_node_to_edge._transform
                to_edge._transform = Transform.identity()

                msg = (
                    f"The to-edge (index: {idx}) for joint '{joint_node.path}' is not set to"
                    " identity. Setting it identity and updating other transforms as necessary."
                )
                graph.logger.debug(msg)
