import os
from xml.etree import ElementTree as ET
import shutil
from .format_coacd import process_single_directory_with_coacd
from pathlib import Path

def indent_xml(elem, level=0):
    """
    Recursive function to indent an XML element
    
    Args:
        elem: XML element to indent
        level: Current indentation level
    """
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for subelem in elem:
            indent_xml(subelem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def find_parent(root, target):
    for parent in root.iter():
        for child in parent:
            if child == target:
                return parent
    return None

def add_collision_to_only_visual_mjcf(mjcf_path, preprocess_resolution=20, resolution=2000, params = None):
    """
    Add collision geometries to MJCF file based on VHACD decomposition
    
    Args:
        mjcf_path: Path to the MJCF file
    """
    # Parse MJCF file using lxml
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    add_collision_to_body_without_collision = False
    if params is not None and params["customer"] == "deepmind":
        add_collision_to_body_without_collision = True
    
    # Get the directory containing the MJCF file
    model_dir = os.path.dirname(mjcf_path)
    
    # Find the visual mesh file path from the MJCF
    asset_elem = root.find('asset')
    if asset_elem is None:
        raise ValueError("No asset element found in MJCF file")
    
    # 第一步：识别需要处理的物体
    bodies_needing_collision = []
    visual_meshes = []
    
    # 找出视觉网格和对应的body
    for body in root.findall('.//body'):
        visual_mesh_name = None
        
        # 检查body是否已经有碰撞网格
        for geom in body.findall('geom'):
            #为每一个visual添加collision
            if geom.get('class') == 'visual' and geom.get('mesh'):
                visual_mesh_name = geom.get('mesh')
                geom_pos = geom.get('pos',[0,0,0])
                geom_euler = geom.get('euler')
                geom_quat = geom.get('quat')
                
            if visual_mesh_name:
                # 获取对应的视觉mesh文件
                for mesh in asset_elem.findall('mesh'):
                    if mesh.get('name') == visual_mesh_name:
                        mesh_file = mesh.get('file', '')
                        if mesh_file:
                            bodies_needing_collision.append((geom_pos, geom_euler, geom_quat, visual_mesh_name))
                            visual_meshes.append(mesh_file)
                            break
    
    if not bodies_needing_collision:
        print("No bodies need collision meshes")
        return
    
    print(f"需要处理的物体数量: {len(bodies_needing_collision)}")
    
    # 第二步：只处理需要的物体
    collision_files_map = {}
    
    # 移除已有的碰撞元素（如果不是保留原有碰撞）
    if not add_collision_to_body_without_collision:
        for elem in root.findall(".//*[@class='collision']"):
            parent = find_parent(root, elem)
            if parent is not None and parent.tag != 'default':
                parent.remove(elem)
    
    # 处理需要碰撞的物体
    print(f"需要处理的物体数量: {bodies_needing_collision}")


    collision_dir = os.path.join(model_dir, "collision")
    if os.path.exists(collision_dir):
        shutil.rmtree(collision_dir)

    os.makedirs(collision_dir)
    
    collision_files_properties = {}
    for geom_pos, geom_euler, geom_quat, visual_mesh_name in bodies_needing_collision:
        # 从visual_mesh_name提取出基础名称（移除_mesh后缀）
        base_name = visual_mesh_name
        if '_mesh' in base_name:
            base_name = base_name.split('_mesh')[0]
        elif '_vis' in base_name:
            base_name = base_name.split('_vis')[0]
        
        print(f"处理物体: {base_name}")
        
        # 生成此物体的碰撞网格
        collision_files = process_single_directory_with_coacd(model_dir, base_name, preprocess_resolution=preprocess_resolution, resolution=resolution)
        
        for collision_file in collision_files:
            collision_file_name = os.path.basename(collision_file)
            print(f"生成子碰撞网格: {collision_file_name}")
            collision_files_properties[collision_file_name] = (geom_pos, geom_euler, geom_quat)
        # if not collision_files:
        #     print(f"未能为物体 {base_name} 生成碰撞网格")
        #     continue
        
        # collision_files_map[base_name] = collision_files
    
    # 确认是否生成了碰撞网格
    collision_dir = os.path.join(model_dir, 'collision')
    if not os.path.exists(collision_dir) or not os.listdir(collision_dir):
        raise ValueError("No collision meshes generated")
    
    # 第三步：将碰撞网格添加到MJCF文件
    all_collision_files = [f for f in os.listdir(collision_dir) if f.endswith('.obj')]
    
    if not all_collision_files:
        raise ValueError("No collision meshes found in collision directory")
    
    # 为每个碰撞文件添加碰撞几何体
    for collision_file in all_collision_files:
        mesh_name = collision_file.replace('.obj', '_mesh')
        collision_prefix = mesh_name.replace('_mesh','')
        
        # 查找对应的body
        object_body = None
        with_collision = False
        
        for body in root.findall('.//body'):
            for geom in body.findall('geom'):
                visual_mesh_name = geom.get('mesh', '')
                if visual_mesh_name and \
                   geom.get('class') == 'visual' and \
                   (visual_mesh_name.split("_mesh")[0] in collision_prefix or 
                    visual_mesh_name.split("_vis")[0] in collision_prefix):
                    object_body = body
                    break
            if object_body is not None:
                break
        
        # 检查是否已有碰撞几何体
        if object_body is not None:
            for geom in object_body.findall('geom'):
                if geom.get('class') == 'collision' and geom.get('type') != 'mesh':
                    with_collision = True
                    break
        
        if with_collision and add_collision_to_body_without_collision: 
            print(f"跳过 {mesh_name} 因为它已有碰撞几何体")
            continue
        
        if object_body is None:
            print(f"警告: 未找到网格 {mesh_name} 对应的body")
            continue
        
        
        
        geom_pos, geom_euler, geom_quat = collision_files_properties.get(collision_file)
        
        # 添加碰撞网格到asset
        collision_mesh = ET.SubElement(asset_elem, 'mesh')
        collision_mesh.set('file', f'collision/{collision_file}')
        collision_mesh.set('name', mesh_name)
        collision_mesh.tail = '\n    '
        
        # 添加碰撞几何体到body
        collision_geom = ET.SubElement(object_body, 'geom')
        collision_geom.set('mesh', mesh_name)
        collision_geom.set('type', 'mesh')
        collision_geom.set('class', 'collision')
        if geom_pos is not None:
            collision_geom.set('pos', geom_pos)
        if geom_euler is not None:
            collision_geom.set('euler', geom_euler)
        if geom_quat is not None:
            collision_geom.set('quat', geom_quat)
        
        collision_geom.tail = '\n        '
    
    # 格式化XML输出
    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        indent_xml(root)
    
    # 保存修改后的MJCF
    tree.write(mjcf_path, encoding='utf-8', xml_declaration=True)
    print(f"成功为MJCF文件添加碰撞几何体: {mjcf_path}")
