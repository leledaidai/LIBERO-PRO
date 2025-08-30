import os
from pxr import Usd, UsdGeom, UsdShade

from lightwheel.srl.from_usd._from_usd_helper import UsdMaterialWrapper

def export_usd_to_obj(usd_file_path, output_dir):
    # Open the USD stage
    stage = Usd.Stage.Open(usd_file_path)
    if not stage:
        print(f"Failed to open USD file: {usd_file_path}")
        return

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all mesh primitives in the USD file
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            mesh_name = prim.GetName()
            
            # Get material subsets
            subsets = UsdGeom.Subset.GetAllGeomSubsets(mesh)
            
            # Check if the entire mesh has a direct material binding
            mesh_binding = UsdShade.MaterialBindingAPI(mesh.GetPrim())
            mesh_material = mesh_binding.ComputeBoundMaterial()[0]
            
            if not subsets and mesh_material:
                # If no subsets but has material, export as single file
                obj_file_path = os.path.join(output_dir, f"{mesh_name}.obj")
                mtl_file_path = os.path.join(output_dir, f"materials.mtl")
                export_mesh_to_obj(mesh, obj_file_path, mesh_material,mesh_material,mtl_file_path)
            else:
                # Export separate OBJ for each material subset
                for subset in subsets:
                    binding = UsdShade.MaterialBindingAPI(subset.GetPrim())
                    material = binding.ComputeBoundMaterial()[0]
                    
                    if material:
                        material_name = material.GetPrim().GetName()
                        obj_file_path = os.path.join(output_dir, f"{mesh_name}_{material_name}.obj")
                        mtl_file_path = os.path.join(output_dir, f"{mesh_name}_{material_name}.mtl")
                        export_mesh_subset_to_obj(mesh, subset, material, obj_file_path, mtl_file_path)

def export_mesh_subset_to_obj(mesh:UsdGeom.Mesh, subset:UsdGeom.Subset, material:UsdShade.Material, obj_file_path:str, mtl_file_path:str):
    """Export a specific material subset of a USD mesh to OBJ/MTL format."""
    # Get mesh data
    points = mesh.GetPointsAttr().Get()
    face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
    face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
    
    # Optimize UV and Normal data retrieval
    normals = None
    uvs = None
    uv_indices = None
    uv_interpolation = None
    normal_indices = None
    normal_interpolation = None
    
    primvar_api = UsdGeom.PrimvarsAPI(mesh)
    #get uvs
    st_primvar = primvar_api.GetPrimvar("st")
    if st_primvar and st_primvar.IsDefined():
        uvs = st_primvar.Get()
        if st_primvar.GetIndicesAttr().IsDefined():
            uv_indices = st_primvar.GetIndicesAttr().Get()
        uv_interpolation = st_primvar.GetInterpolation()
        
    #get normals
    normals_primvar = primvar_api.GetPrimvar("normals")
    if normals_primvar and normals_primvar.IsDefined():
        normals = normals_primvar.Get()
        if normals_primvar.GetIndicesAttr().IsDefined():
            normal_indices = normals_primvar.GetIndicesAttr().Get()
        normal_interpolation = normals_primvar.GetInterpolation()
    # else:
    #     normals = mesh.GetNormalsAttr().Get()

 
    # Get faces for this subset
    subset_faces = subset.GetIndicesAttr().Get()
    
    # Write OBJ file
    with open(obj_file_path, 'w') as obj_file:
        obj_file.write(f"mtllib {os.path.basename(mtl_file_path)}\n")
        obj_file.write(f"o {mesh.GetPrim().GetName()}_{material.GetPrim().GetName()}\n\n")
        obj_file.write(f"usemtl {material.GetPrim().GetName()}\n")

        # Write vertices, UVs, and normals
        for point in points:
            obj_file.write(f"v {point[0]} {point[1]} {point[2]}\n")
        if uvs:
            for uv in uvs:
                obj_file.write(f"vt {uv[0]} {uv[1]}\n")
        if normals:
            for normal in normals:
                obj_file.write(f"vn {normal[0]} {normal[1]} {normal[2]}\n")

        
        vertex_offset = 0
        face_to_vertex_offset = {}
        
        # Build face offset mapping
        for face_idx, count in enumerate(face_vertex_counts):
            face_to_vertex_offset[face_idx] = vertex_offset
            vertex_offset += count

        # Write only the faces in this subset
        write_faces(obj_file, subset_faces, face_vertex_counts, face_vertex_indices,
                   face_to_vertex_offset, uv_indices, normal_indices,uvs is not None, normals is not None,uv_interpolation,normal_interpolation)

    usd_material_info = UsdMaterialWrapper.from_usd(material)
    export_usd_material_to_mtl(usd_material_info, mtl_file_path)
    
    return usd_material_info

def export_mesh_to_obj(mesh, obj_file_path, mesh_material,mtl_file_path):
    """Export USD mesh without mesh subsets to OBJ/MTL format."""
    # Get mesh data
    points = mesh.GetPointsAttr().Get()
    face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
    face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
    
    # Optimize UV data retrieval
    uvs = None
    uv_indices = None
    uv_interpolation = None
    
    normals = None
    normal_indices = None
    normal_interpolation = None
    primvar_api = UsdGeom.PrimvarsAPI(mesh)
    #get normals
    normals_primvar = primvar_api.GetPrimvar("normals")
    if normals_primvar and normals_primvar.IsDefined():
        normals = normals_primvar.Get()
        normal_interpolation = normals_primvar.GetInterpolation()
        if normals_primvar.GetIndicesAttr().IsDefined():
            normal_indices = normals_primvar.GetIndicesAttr().Get()
    #直接从mesh获取法线的效果其差故而抛弃
    # else:
    #     normals = mesh.GetNormalsAttr().Get() 
    #normals = mesh.GetNormalsAttr().Get() 
    #get uvs
    st_primvar = primvar_api.GetPrimvar("st")
    if st_primvar and st_primvar.IsDefined():
        uvs = st_primvar.Get()
        uv_interpolation = st_primvar.GetInterpolation()
        if st_primvar.GetIndicesAttr().IsDefined():
            uv_indices = st_primvar.GetIndicesAttr().Get()


    # Get material subsets
    default_faces = set(range(len(face_vertex_counts)))  # Track faces not in any subset
    
    # Check if the entire mesh has a direct material binding
    mesh_binding = UsdShade.MaterialBindingAPI(mesh.GetPrim())
    mesh_material = mesh_binding.ComputeBoundMaterial()[0]
    usd_material_info = None
    material_name = None
    if mesh_material and mesh_material.GetPrim():
        # If the mesh has a direct material binding, use it as the default material
        material_name = mesh_material.GetPrim().GetName()
        usd_material_info = UsdMaterialWrapper.from_usd(mesh_material)
        export_usd_material_to_mtl(usd_material_info, mtl_file_path)
        
    
    # Write OBJ file
    with open(obj_file_path, 'w') as obj_file:
        obj_file.write(f"mtllib {os.path.basename(mtl_file_path)}\n")
        if material_name:
            obj_file.write(f"usemtl {mesh_material.GetPrim().GetName()}\n")
        obj_file.write(f"o {mesh.GetPrim().GetName()}\n\n")

        # Write vertices
        for point in points:
            obj_file.write(f"v {point[0]} {point[1]} {point[2]}\n")

        # Write texture coordinates
        if uvs:
            for uv in uvs:
                obj_file.write(f"vt {uv[0]} {uv[1]}\n")
        # Write normals
        if normals:
            for normal in normals:
                obj_file.write(f"vn {normal[0]} {normal[1]} {normal[2]}\n")
                
        # Write faces grouped by materials
        vertex_offset = 0
        face_to_vertex_offset = {}  # Map face index to its vertex offset
        
        # Build face offset mapping
        for face_idx, count in enumerate(face_vertex_counts):
            face_to_vertex_offset[face_idx] = vertex_offset
            vertex_offset += count

        # # Write faces for each material subset
        # for material_name, face_indices in material_subsets.items():
        #     obj_file.write(f"usemtl {material_name}\n")
        #     write_faces(obj_file, face_indices, face_vertex_counts, face_vertex_indices, 
        #                face_to_vertex_offset, uv_indices, uvs is not None, normals is not None)

        #Write remaining faces with mesh material or default material
        if default_faces:
            obj_file.write(f"usemtl {material_name}\n")  # Using the material_name we determined earlier
            write_faces(obj_file, default_faces, face_vertex_counts, face_vertex_indices,
                       face_to_vertex_offset, uv_indices, normal_indices, uvs is not None, normals is not None,uv_interpolation,normal_interpolation)
    
    
    
    return usd_material_info

def write_faces(obj_file, face_indices, face_vertex_counts, face_vertex_indices, 
                face_to_vertex_offset, uv_indices, normal_indices,has_uvs, has_normals,uv_interpolation,normal_interpolation):
    """Helper function to write faces for a given material group."""
    try:     
        for face_idx in face_indices:
        
            vertex_offset = face_to_vertex_offset[face_idx]
            count = face_vertex_counts[face_idx]
            indices = face_vertex_indices[vertex_offset:vertex_offset + count]
            face_line = "f"
            
            normal_indice = None
            uv_indice = None
            
            if uv_indices and uv_interpolation == "faceVarying":  
                uv_indice = uv_indices[vertex_offset:vertex_offset + count]
            if normal_indices and normal_interpolation == "faceVarying":
                normal_indice = normal_indices[vertex_offset:vertex_offset + count]
            
            for i,idx in enumerate(indices):
                uv_idx = None
                normal_idx = None
                
                if has_uvs and uv_interpolation == "vertex" and uv_indices is not None:
                    uv_idx = uv_indices[idx]
                elif has_uvs and uv_interpolation == "vertex" and uv_indices is None:
                    uv_idx = idx
                elif has_uvs and uv_interpolation == "faceVarying" and uv_indices is not None:
                    uv_idx = uv_indice[i]
                elif has_uvs:
                    uv_idx = face_idx*3+i
                  
                if has_normals and normal_interpolation == "vertex" and normal_indices is not None:
                    normal_idx = normal_indices[idx]
                elif has_normals and normal_interpolation == "vertex" and normal_indices is None:
                    normal_idx = idx
                elif has_normals and normal_interpolation == "faceVarying" and normal_indices is not None:
                    normal_idx = normal_indice[i]
                elif has_normals:
                    normal_idx = face_idx*3+i
                
                v = idx
                vt = None
                vn = None
                
                if uv_idx is not None:
                    vt = uv_idx
                elif has_uvs and uv_idx is None:
                    vt = idx
                
                if normal_idx is not None:
                    vn = normal_idx
                elif has_normals and normal_idx is None:
                    vn = idx
                
                # OBJ indices are 1-based 
                if vt is not None and vn is not None:
                    face_line += f" {v+1}/{vt+1}/{vn+1}"
                elif vt is not None:
                    face_line += f" {v+1}/{vt+1}"
                elif vn is not None:
                    face_line += f" {v+1}//{vn+1}"
                else:
                    face_line += f" {v+1}"
                    
            obj_file.write(face_line + "\n")
    
    except ValueError as e:
        print(e)
        return
       
def export_usd_material_to_mtl(material:UsdMaterialWrapper, mtl_file_path:str):
    """Export USD material to MTL format."""
    material_name = material.material_name
        
    #Reset .mtl file
    if not hasattr(export_usd_material_to_mtl, 'exported_materials'):
        if os.path.exists(mtl_file_path):
            os.remove(mtl_file_path)
        export_usd_material_to_mtl.exported_materials = set()
        
    if hasattr(export_usd_material_to_mtl, 'exported_materials'):
        # Check if material already exported
        if material_name in export_usd_material_to_mtl.exported_materials:
            return
        
        material.export_mtl(mtl_file_path)
        export_usd_material_to_mtl.exported_materials.add(material_name)

# 主函数
if __name__ == "__main__":

    usd_file = "/media/lightwheel/Data3/Content/bottle/Bottle019/Bottle019.usd"
    output_dir = "/media/lightwheel/Data3/Content/bottle/Bottle019/Bottle019_obj"
    export_usd_to_obj(usd_file, output_dir)