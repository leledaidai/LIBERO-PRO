# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Private helper functions for the `from_usd` package."""

# Standard Library
from typing import Optional
import os

# Third Party
import numpy as np
from pxr import Usd, UsdGeom,UsdShade

# NVIDIA
import lightwheel.srl.usd.prim_helper as prim_helper
from lightwheel.srl.basics.enum import Enum, auto


class NodeType(Enum):
    """Enum to denote the node type."""

    GEOMETRY = auto()
    JOINT = auto()
    LINK = auto()
    SENSOR = auto()
    PHONY = auto()
    
class MaterialType(Enum):
    UsdPreviewSurface = "UsdPreviewSurface"
    OmniPBR = "OmniPBR"
    UnSupportedMaterialType = "UnSupportedMaterialType"

class UsdMaterialWrapper:
    def __init__(self, material_name:str, material_type:MaterialType, material_properties:dict):
        self.material_name = material_name
        self.material_type = material_type
        self.material_properties = material_properties
        
    @classmethod
    def from_usd(cls,material:UsdShade.Material):
        material_name = material.GetPrim().GetName()
        materail_type = UsdMaterialWrapper.get_materail_type(material)
        
        if materail_type == MaterialType.UsdPreviewSurface.value:
            material_properties = UsdMaterialWrapper.get_UsdPreviewSurface_properties(material)
            
            texture_path = UsdMaterialWrapper.get_material_texture(material,'diffuseColor')
            if texture_path:
                material_properties['diffuse_texture'] = texture_path
            
            normal_path = UsdMaterialWrapper.get_material_texture(material,'normal')
            if normal_path:
                material_properties['normal_texture'] = normal_path
            
            return UsdMaterialWrapper(material_name,MaterialType.UsdPreviewSurface,material_properties)
        elif materail_type == MaterialType.OmniPBR.value:
            material_properties = UsdMaterialWrapper.get_OmniPBR_properties(material)
            return UsdMaterialWrapper(material_name,MaterialType.OmniPBR,material_properties)
        else:
            return UsdMaterialWrapper(material_name,MaterialType.UnSupportedMaterialType,None)
        
    @classmethod
    def get_materail_type(cls,material:UsdShade.Material)->str:
        surface = material.GetSurfaceOutput("mdl")
        if not surface:
            return "UsdPreviewSurface"

        connected = surface.GetConnectedSource()
        if not connected:
            return "Unconnected MDL"

        shader = UsdShade.Shader(connected[0])

        # 确保它是 MDL shader,暂不清楚为什么部分 mdl shader没有GetIdAttr
        # shader_id = shader.GetIdAttr().Get()
        # if shader_id != "mdl":
        #     return "Not MDL Shader"

        # 获取 subIdentifier
        sub_id_attr = shader.GetPrim().GetAttribute("info:mdl:sourceAsset:subIdentifier")
        if not sub_id_attr or not sub_id_attr.HasAuthoredValue():
            return "MDL Shader Without SubIdentifier"

        mdl_type = sub_id_attr.Get()
        return mdl_type

    @classmethod
    def get_UsdPreviewSurface_properties(cls,material:UsdShade.Material)->dict:
        property_dict = {}
        shader, sourceName, sourceType = material.ComputeSurfaceSource(renderContext="Default")
        if not shader:
            return None
        inputs = shader.GetInputs()
        #TODO: 将纹理读取和颜色读取整合在此
        for input in inputs:
            input_name = input.GetFullName()
            input_value = input.Get()
            property_dict[input_name] = input_value
        return property_dict
    
    @classmethod
    def get_OmniPBR_properties(cls,material:UsdShade.Material)->dict:
        property_dict = {}
        shader, sourceName, sourceType = material.ComputeSurfaceSource(renderContext="mdl")
        if not shader:
            return None
        inputs = shader.GetInputs()
        for input in inputs:
            input_name = input.GetFullName()
            input_value = input.Get()
            property_dict[input_name] = input_value
        return property_dict

    def export_mtl(self,mtl_file_path):
        
        #textures_output_dir = os.path.join(mtl_file_path.parent)
        textures_output_dir = os.path.join(mtl_file_path.parent,'textures')
        source_path = os.path.join(mtl_file_path.parent)
        
        if self.material_type == MaterialType.UsdPreviewSurface:
            with open(mtl_file_path, 'a') as mtl_file:
                mtl_file.write(f"\nnewmtl {self.material_name}\n")
                
                if self.material_properties.get('inputs:diffuseColor'):
                    Kd = self.material_properties.get('inputs:diffuseColor')
                    mtl_file.write(f"Kd {Kd[0]} {Kd[1]} {Kd[2]}\n")
                    
                # if self.material_properties.get('inputs:emissiveColor'):
                #     Ke = self.material_properties.get('inputs:emissiveColor')
                #     mtl_file.write(f"Ke {Ke[0]} {Ke[1]} {Ke[2]}\n")
                    
                if self.material_properties.get('inputs:opacity'):
                    opacity = self.material_properties.get('inputs:opacity')
                    mtl_file.write(f"d {opacity}\n")
                    
                if self.material_properties.get('diffuse_texture'):
                    old_texture_path = self.material_properties.get('diffuse_texture')
                    new_texture_path = export_texture(old_texture_path,textures_output_dir)
                    if new_texture_path:
                        relative_path = os.path.relpath(new_texture_path, source_path)
                        relative_path = os.path.join(os.sep, relative_path)
                        mtl_file.write(f"map_Kd {relative_path}\n")
                                            
                if self.material_properties.get('normal_texture'):
                    old_texture_path = self.material_properties.get('normal_texture')
                    new_texture_path = export_texture(old_texture_path,textures_output_dir)
                    if new_texture_path:
                        relative_path = os.path.relpath(new_texture_path, source_path)
                        relative_path = os.path.join(os.sep, relative_path)
                        mtl_file.write(f"map_Bump {relative_path}\n")
            
        elif self.material_type == MaterialType.OmniPBR:
            with open(mtl_file_path, 'a') as mtl_file:
                mtl_file.write(f"\nnewmtl {self.material_name}\n")
                
                if self.material_properties.get('inputs:diffuse_color_constant'):
                    Kd = self.material_properties.get('inputs:diffuse_color_constant')
                    mtl_file.write(f"Kd {Kd[0]} {Kd[1]} {Kd[2]}\n")
                    
                # if self.material_properties.get('inputs:emissive_color'):
                #     Ke = self.material_properties.get('inputs:emissive_color')
                #     mtl_file.write(f"Ke {Ke[0]} {Ke[1]} {Ke[2]}\n")
                    
                # if self.material_properties.get('inputs:opacity_constant'):
                #     opacity = self.material_properties.get('inputs:opacity_constant')
                #     mtl_file.write(f"d {opacity}\n")
                    
                if self.material_properties.get('inputs:diffuse_texture'):
                    old_texture_path = self.material_properties.get('inputs:diffuse_texture').resolvedPath
                    new_texture_path = export_texture(old_texture_path,textures_output_dir)
                    if new_texture_path:
                        relative_path = os.path.relpath(new_texture_path, source_path)
                        relative_path = os.path.join(os.sep, relative_path)
                        mtl_file.write(f"map_Kd {relative_path}\n")
                    
                # if self.material_properties.get('inputs:metallic_texture'):
                #     old_texture_path = self.material_properties.get('inputs:metallic_texture').resolvedPath
                #     new_texture_path = export_texture(old_texture_path,textures_output_dir)
                #     if new_texture_path:
                #         relative_path = os.path.relpath(new_texture_path, source_path)
                #         relative_path = os.path.join(os.sep, relative_path)
                #         mtl_file.write(f"map_Ks {relative_path}\n")
                        
                if self.material_properties.get('inputs:normalmap_texture'):
                    old_texture_path = self.material_properties.get('inputs:normalmap_texture').resolvedPath
                    new_texture_path = export_texture(old_texture_path,textures_output_dir)
                    if new_texture_path:
                        relative_path = os.path.relpath(new_texture_path, source_path)
                        relative_path = os.path.join(os.sep, relative_path)
                        mtl_file.write(f"map_Bump {relative_path}\n")
            
    def export_mjcf_material(self)->dict:
        #mjcf material properties        
    
        #rgba Color and transparency of the material. All components should be in the range [0 1].
        #reflectance This attribute should be in the range [0 1].
        #emission mjcf only provide a scalar setting. 
        #specular mjcf only provide a scalar setting.
        #shininess one float value The value given here is multiplied by 128 before passing it to OpenGL, so it should be in the range [0 1].
    
        #diffuse_texture mjcf only support diffuse texture
        
        mjcf_material ={}
        
        if self.material_type == MaterialType.UsdPreviewSurface:
            if self.material_properties.get('inputs:diffuseColor') and self.material_properties.get('inputs:opacity'):
                mjcf_material['rgba'] = list(self.material_properties.get('inputs:diffuseColor'))+[self.material_properties.get('inputs:opacity')]
            if self.material_properties.get('inputs:diffuseColor'):
                mjcf_material['rgba'] = list(self.material_properties.get('inputs:diffuseColor'))+[1.0]
            # if self.material_properties.get('inputs:specular_level'):
            #     mjcf_material['specular'] = self.material_properties.get('inputs:specular_level')
            if self.material_properties.get('diffuse_texture'):
                texture_path = self.material_properties.get('diffuse_texture')
                texture_name = os.path.basename(texture_path)
                mjcf_material['diffuse_texture'] = texture_name
        elif self.material_type == MaterialType.OmniPBR:
            if self.material_properties.get('inputs:diffuse_color_constant') and self.material_properties.get('inputs:opacity_constant'):
                mjcf_material['rgba'] = list(self.material_properties.get('inputs:diffuse_color_constant'))+[self.material_properties.get('inputs:opacity_constant')]
            if self.material_properties.get('inputs:diffuse_color_constant'):
                mjcf_material['rgba'] = list(self.material_properties.get('inputs:diffuse_color_constant'))+[1.0]
            # if self.material_properties.get('inputs:specular_level'):
            #     mjcf_material['specular'] = self.material_properties.get('inputs:specular_level')
            if self.material_properties.get('inputs:diffuse_texture'):
                texture_path = self.material_properties.get('inputs:diffuse_texture').resolvedPath
                texture_name = os.path.basename(texture_path)
                mjcf_material['diffuse_texture'] = texture_name
            
        return mjcf_material
    
    def get_urdf_material(self)->dict:
        #urdf material properties        
    
        #support exporting rgba now        
        urdf_material ={}
        
        if self.material_type == MaterialType.UsdPreviewSurface:
            if self.material_properties.get('inputs:diffuseColor') and self.material_properties.get('inputs:opacity'):
                urdf_material['rgba'] = list(self.material_properties.get('inputs:diffuseColor'))+[self.material_properties.get('inputs:opacity')]
            if self.material_properties.get('inputs:diffuseColor'):
                urdf_material['rgba'] = list(self.material_properties.get('inputs:diffuseColor'))+[1.0]
            # if self.material_properties.get('inputs:specular_level'):
            #     mjcf_material['specular'] = self.material_properties.get('inputs:specular_level')
        elif self.material_type == MaterialType.OmniPBR:
            if self.material_properties.get('inputs:diffuse_color_constant') and self.material_properties.get('inputs:opacity_constant'):
                urdf_material['rgba'] = list(self.material_properties.get('inputs:diffuse_color_constant'))+[self.material_properties.get('inputs:opacity_constant')]
            if self.material_properties.get('inputs:diffuse_color_constant'):
                urdf_material['rgba'] = list(self.material_properties.get('inputs:diffuse_color_constant'))+[1.0]
            # if self.material_properties.get('inputs:specular_level'):
            #     mjcf_material['specular'] = self.material_properties.get('inputs:specular_level')

            
        return urdf_material
    
    @classmethod
    def get_material_texture(cls,material:UsdShade.Material, input_name:str):

        #try:
            #try to get the usdpreviewsurface shader texture
            shader, sourceName, sourceType = material.ComputeSurfaceSource(renderContext="Default")
            if shader:
                
                texture_property = input_name
                input = shader.GetInput(texture_property)
                texture_path = traverse_nodegraph2(input,texture_property)
                if texture_path:
                    return texture_path.resolvedPath
                else:
                    return None
        
            #try to get the mdl shader texture
            shader, sourceName, sourceType = material.ComputeSurfaceSource(renderContext="mdl")
            if shader:
                texture_property = input_name
                input = shader.GetInput(texture_property)
                texture_path = input.Get()
                if texture_path:
                    return texture_path.resolvedPath
                else:
                    return None
                
            return None
        # except Exception as e:
        #       print(f"Error getting texture path for {material.GetPath()}: {e}")
        #       return None

    def isValid(self) -> bool:
        if not self.material_properties:
            return False
        else:
            return True
        
    
    def __repr__(self):
        return f"MaterialInfo(usd_material={self.usd_material}, material_type={self.material_type}, material_properties={self.material_properties})"

def _is_geometry_prim(prim: Usd.Prim) -> bool:
    """Check if the given prim is a primitive geometry object prim."""
    geometry_types = [UsdGeom.Cylinder, UsdGeom.Cube, UsdGeom.Sphere, UsdGeom.Mesh]

    for geometry_type in geometry_types:
        if prim.IsA(geometry_type):
            return True
    return False


def _is_joint_prim(prim: Usd.Prim) -> bool:
    """Check if the given prim is a joint prim."""
    return prim_helper.is_a_joint(prim)


def _is_link_prim(prim: Usd.Prim) -> bool:
    """Check if the given prim is a link prim."""
    # A prim without a type is assumed to be of type Xform with an identity transform
    if prim.GetTypeName() == "":
        return True
    return prim.IsA(UsdGeom.Xform)


def _is_sensor_prim(prim: Usd.Prim) -> bool:
    """Check if the given prim is a sensor object prim."""
    valid_type_names = ["Camera", "IsaacImuSensor"]
    return prim.GetTypeName() in valid_type_names


def _is_urdf_prim(prim: Usd.Prim) -> bool:
    """Check if the given prim is a prim used in the URDF."""
    return _is_joint_prim(prim) or _is_link_prim(prim) or _is_geometry_prim(prim)


def _get_prim_scale(prim: Usd.Prim) -> Optional[np.ndarray]:
    """Get the prim's scaling value."""
    try:
        scale = prim.GetAttribute("xformOp:scale").Get()
        if scale is not None:
            return np.array(scale)
        else:
            print("Scale not exists,set to:  ",(1,1,1))
            return np.array((1,1,1))
    except ValueError:
        return None

def export_texture(texture_path,output_dir):
    if not os.path.exists(texture_path):
        print(f"Failed to export texture file: Can not find {texture_path}")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_name = os.path.basename(texture_path)
    target_path = os.path.join(output_dir,file_name)
    
    try:
        if os.path.exists(texture_path):
            new_texture_name = os.path.basename(texture_path)
            texture_name, ext = os.path.splitext(new_texture_name)
            if ext == '.jpg' or ext == '.JPG':
                from PIL import Image
                img = Image.open(texture_path)
                target_path = target_path.replace('.jpg','.png')
                target_path = target_path.replace('.JPG','.png')
                img.save(target_path,format="PNG", compress_level=0)
                print(f"Exported texture file: {texture_path} to {target_path}")
            if ext == '.png' or ext == '.PNG':
                import shutil
                shutil.copy(texture_path,target_path)
                print(f"Exported texture file: {texture_path} to {target_path}")
        return target_path
    except Exception as e:
        print(f"Failed to export texture file: {e}")
        return None
    
def traverse_nodegraph2(input,input_name):
        if not input:
            return
        #如果diffuseColor链接的是颜色数值而不是纹理返回空
        # input_value = input.Get()
        # if input_value is not None:
        #     return
        source= UsdShade.ConnectableAPI.GetConnectedSource(input)
        if not source:
            return
        if source[0].GetPrim().IsA('NodeGraph'):
            if source[2] == UsdShade.AttributeType.Output:
                nodegraph = UsdShade.NodeGraph(source[0].GetPrim())
                outputs = nodegraph.GetOutput(source[1])
                file_path = traverse_nodegraph2(outputs,input_name)
                return file_path
        if source[0].GetPrim().IsA('Shader'):
            shader = UsdShade.Shader(source[0].GetPrim())
            file_input = shader.GetInput('file')
            if file_input:
                file_path = file_input.Get()
                return file_path