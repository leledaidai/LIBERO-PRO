# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""USD to Graphviz CLI."""

# Standard Library
import argparse
from pathlib import Path
from typing import List, Optional

# NVIDIA
import lightwheel.srl.from_usd.transform_graph_tools as transform_graph_tools
import lightwheel.srl.tools.logger as logger
from lightwheel.srl.from_usd.transform_graph import TransformEdge, TransformGraph, TransformNode


def main() -> None:
    """Run usd_to_graphviz CLI."""
    parser = argparse.ArgumentParser("USD to Graphviz.")

    parser.add_argument("input", type=str, help="The input file path.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help=(
            "The output directory. If not provided the files will be saved to the current working"
            " directory."
        ),
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default=None,
        help="Dot and PNG file name (without the extension). Defaults to TransformGraph name.",
    )
    parser.add_argument(
        "-u",
        "--reduce-to-urdf-tree",
        action="store_true",
        help=(
            "If set, attempt to try to reduce the Transform Graph to a URDF compliant tree. The"
            " resulting files are saved in the same directory as the full graph with `_urdf` append"
            " to the filename."
        ),
    )
    parser.add_argument(
        "-r",
        "--root",
        type=str,
        default=None,
        help=(
            "The root of the robot's kinematic tree. Can either be specified as with the prim path"
            " or with the TransformGraph node name. Defaults to the path of the stage's default"
            " prim. Only applicable if the `--reduce-to_urdf-tree` option is set."
        ),
    )
    parser.add_argument(
        "--remove-nodes",
        nargs="+",
        help=(
            "Names of nodes that should be removed from the TransformGraph so that a valid URDF"
            " TransformGraph can be made. Only applicable if the `--reduce-to_urdf-tree` option is"
            " set."
        ),
    )
    parser.add_argument(
        "--remove-edges",
        nargs="+",
        help=(
            "Names of edges that should be removed from the TransformGraph so that a valid URDF"
            " TransformGraph can be made. Only applicable if the `--reduce-to_urdf-tree` option is"
            " set."
        ),
    )
    parser.add_argument(
        "--parent-link-is-body-1",
        nargs="+",
        help=(
            "Names of joint nodes where the parent link is assumed to be the body 1 target prim,"
            " instead of the default body 0 target prim. Note, when only one body target is set,"
            " then the parent link is assumed to be the default prim, and the child link is the"
            " prim of the set body target."
        ),
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="If set, nothing will be printed to the terminal.",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        type=str,
        choices=["critical", "error", "warning", "info", "debug"],
        default="info",
        help="Set the log level.",
    )

    args = parser.parse_args()

    log_level = logger.level_from_name(args.log_level.upper())

    # Create full graph
    graph = TransformGraph.init_from_usd_path(
        args.input,
        parent_link_is_body_1=args.parent_link_is_body_1,
        log_level=log_level,
    )

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path.cwd()

    # Set saving kwargs for full graph
    kwargs = {
        "output_dir": output_dir,
        "name": args.name,
        "quiet": args.quiet,
    }

    # Save full graph
    graph.save_graphviz(**kwargs)
    graph.render_graphviz(**kwargs)

    if args.reduce_to_urdf_tree:
        # Set reduce_to_urdf args
        nodes_to_remove: Optional[List[TransformNode]] = None
        if args.remove_nodes is not None:
            nodes_to_remove = list(map(lambda name_: graph.get_node(name_), args.remove_nodes))

        edges_to_remove: Optional[List[TransformEdge]] = None
        if args.remove_edges is not None:
            edges_to_remove = list(map(lambda name_: graph.get_edge(name_), args.remove_edges))
        if args.root is None:
            root_node = None
        else:
            root_node = graph.get_node(args.root)

        # Try to reduce graph to URDF tree
        transform_graph_tools.reduce_to_urdf(
            graph,
            nodes_to_remove=nodes_to_remove,
            edges_to_remove=edges_to_remove,
            root_node=root_node,
        )

        # Set saving kwargs for reduced graph
        if args.name is None:
            name = None
        else:
            name = f"{args.name}_urdf"
        kwargs = {
            "output_dir": args.output_dir,
            "name": name,
            "quiet": args.quiet,
        }

        # Save reduced graph
        graph.save_graphviz(**kwargs)
        graph.render_graphviz(**kwargs)


if __name__ == "__main__":
    main()
