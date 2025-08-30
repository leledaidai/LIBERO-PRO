import os
import re
import numpy as np
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import array_to_string
 
import pathlib
 
# 使用pathlib计算项目根目录的绝对路径
absolute_path = pathlib.Path(__file__).parent.parent.parent.absolute()
 
from libero.libero.envs.base_object import register_object
 
 
class CustomObjects(MujocoXMLObject):
    def __init__(self, custom_path, name, obj_name, joints=[dict(type="free", damping="0.0005")]):
        # 确保custom_path是一个绝对路径
        assert (os.path.isabs(custom_path)), "Custom path must be an absolute path"
        # 确保custom_path指向一个xml文件
        assert (custom_path.endswith(".xml")), "Custom path must be an xml file"
        super().__init__(
            custom_path,
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()
        self.object_properties = {"vis_site_names": {}}
 
 
@register_object
class LiberoMug(CustomObjects):
    def __init__(self,
                 name="libero_mug",
                 obj_name="libero_mug",
                 ):
        # 使用absolute_path构建正确的绝对路径
        custom_path = "/home/ps/BEHAVIOR-1K/OmniGibson/omnigibson/data/og_dataset/objects/bottle_of_gin/qzgcdx/usd/MJCF/try.xml"
 
        # 打印路径用于调试
        #print(f"构建的路径: {custom_path}")
 
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
 
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None
        
@register_object
class LiberoMugYellow(CustomObjects):
    def __init__(self,
                 name="libero_mug",
                 obj_name="libero_mug",
                 ):
        super().__init__(
            # custom_path=os.path.abspath(os.path.join(
            #     "./", "custom_assets", "libero_mug_yellow", "libero_mug_yellow.xml"
            # )),
            custom_path="/home/ps/BEHAVIOR-1K/OmniGibson/omnigibson/data/og_dataset/objects/bottle_of_gin/qzgcdx/usd/MJCF/try.xml",
            name=name,
            obj_name=obj_name,
        )

        self.rotation = {
            "x": (-np.pi/2, -np.pi/2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None
