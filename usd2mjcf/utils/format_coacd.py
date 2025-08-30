#!/usr/bin/env python3
import os
import numpy as np
import trimesh
import coacd

def process_mesh(input_file, output_dir, threshold=0.01, max_convex_hull=-1, preprocess_mode="auto",
                 preprocess_resolution=20, resolution=2000, mcts_nodes=20, mcts_iterations=150,
                 mcts_max_depth=3, pca=False, no_merge=False, decimate=False, max_ch_vertex=256,
                 extrude=False, extrude_margin=0.01, apx_mode="ch", seed=0, quiet=False):
    if not os.path.isfile(input_file):
        print(input_file, "is not a file")
        exit(1)

    if quiet:
        coacd.set_log_level("error")

    mesh = trimesh.load(input_file, force="mesh")
    mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    result = coacd.run_coacd(
        mesh,
        threshold=threshold,
        max_convex_hull=max_convex_hull,
        preprocess_mode=preprocess_mode,
        preprocess_resolution=preprocess_resolution,
        resolution=resolution,
        mcts_nodes=mcts_nodes,
        mcts_iterations=mcts_iterations,
        mcts_max_depth=mcts_max_depth,
        pca=pca,
        merge=not no_merge,
        decimate=decimate,
        max_ch_vertex=max_ch_vertex,
        extrude=extrude,
        extrude_margin=extrude_margin,
        apx_mode=apx_mode,
        seed=seed,
    )
    output_dir = output_dir  # 获取输出文件的目录
    base_name = os.path.splitext(os.path.basename(input_file))[0]  # 获取输入文件的基础名称
    part_filenames = []
    for i, (vs, fs) in enumerate(result):  # 添加索引 i
        # 检查顶点数量，如果少于4个顶点则跳过
        if len(vs) < 4:
            print(f"跳过网格部分 {base_name}{i+1:03d}，顶点数量({len(vs)})小于4")
            continue
        mesh_part = trimesh.Trimesh(vs, fs)
        if not mesh_part.is_watertight:
            print(f"跳过网格部分 {base_name}{i+1:03d}，不是封闭的")
            continue
        if mesh_part.bounding_box.volume < 1e-10:
            print(f"跳过网格部分 {base_name}{i+1:03d}，边界框体积小于1e-10")
            continue
        mesh_part = trimesh.Trimesh(vs, fs)
        part_filename = os.path.join(output_dir, f"{base_name}{i+1:03d}.obj")  # 生成每个部分的文件名
        mesh_part.export(part_filename)  # 保存每个部分
        part_filenames.append(part_filename)
    return part_filenames


def process_single_directory_with_coacd(style_dir, base_name=None, preprocess_resolution=20, resolution=2000):
    """
    Process a single object directory and perform VHACD decomposition
    
    Args:
        style_dir (str): Directory path containing 'visual'/'visuals' and 'collision' folders.
                        The function will look for .obj mesh files in the visual directory.
        base_name (str, optional): Filter to process only mesh files containing this name.
                                 If None, processes all .obj files found. Defaults to None.
        preprocess_resolution (int, optional): Voxelization resolution for preprocessing stage.
                                             Higher values = more detailed preprocessing but slower.
                                             Defaults to 20.
        resolution (int, optional): Main voxelization resolution for convex decomposition.
                                   Higher values = more accurate decomposition but slower processing.
                                   Defaults to 2000.
    
    Returns:
        list: List of generated collision mesh filenames, or None if no mesh processed.
        
    Raises:
        FileNotFoundError: If neither 'visual' nor 'visuals' directory exists in style_dir.
    """
    collision_dir = os.path.join(style_dir, "collision")
    
    # Process mesh files
    meshes_dir = os.path.join(style_dir, "visual")
    if not os.path.exists(meshes_dir):
        # Try alternate directory name "visuals"
        meshes_dir = os.path.join(style_dir, "visuals")
        if not os.path.exists(meshes_dir):
            raise FileNotFoundError(f"Neither 'visual' nor 'visuals' directory found in {style_dir}")

    for mesh_file in os.listdir(meshes_dir):
        if mesh_file.lower().endswith('.obj') and (base_name is None or base_name in mesh_file):
            input_mesh_path = os.path.join(meshes_dir, mesh_file)  # 获取完整的输入文件路径
            part_filenames = process_mesh(input_mesh_path, collision_dir, preprocess_resolution=preprocess_resolution, resolution=resolution)  # 调用 process_mesh 处理每个网格文件
            return part_filenames  