from robosuite.utils.mjcf_utils import new_site
from libero.libero.envs.bddl_base_domain import BDDLBaseDomain, register_problem
from libero.libero.envs.robots import *
from libero.libero.envs.objects import *
from libero.libero.envs.predicates import *
from libero.libero.envs.regions import *
from libero.libero.envs.utils import rectangle2xyrange

from scipy.spatial.transform import Rotation

def rotate_around_y(original_quat=None, original_pos=None, degrees=0):
    """
    计算四元数和/或3D位置点绕自定义轴 (x=0, z=0.8) 的平行于Y轴的轴旋转指定角度后的新值
    
    参数:
        original_quat: 原始四元数 [w, x, y, z] (可选)
        original_pos: 原始位置 [x, y, z] (可选)
        degrees: 旋转角度（度数），正值为从X轴向Z轴旋转方向
        
    返回:
        字典包含:
        - 'new_quat': 旋转后的四元数 [w, x, y, z] (如果输入了original_quat)
        - 'new_pos': 旋转后的位置 [x, y, z] (如果输入了original_pos)
    """
    result = {}
    
    # 定义旋转轴 (x=0, z=0.8) 的平行于Y轴的向量
    axis = np.array([0, 1, 0])  # 方向与Y轴相同
    axis_point = np.array([0, 0, 0.8])  # 轴经过的点
    
    # 创建绕自定义轴的旋转
    custom_rotation = Rotation.from_rotvec(np.radians(-degrees) * axis)
    
    # 处理四元数旋转
    if original_quat is not None:
        original_rot = Rotation.from_quat([original_quat[1], original_quat[2], original_quat[3], original_quat[0]])
        combined_rot = custom_rotation * original_rot
        new_quat = combined_rot.as_quat()
        result['new_quat'] = [float(new_quat[3]), float(new_quat[0]), float(new_quat[1]), float(new_quat[2])]
    
    # 处理位置点旋转
    if original_pos is not None:
        # 对于点旋转，需要先平移到旋转轴，旋转后再平移回来
        translated_pos = np.array(original_pos) - axis_point
        rotated_pos = custom_rotation.apply(translated_pos)
        final_pos = rotated_pos + axis_point
        result['new_pos'] = final_pos.tolist() if isinstance(final_pos, np.ndarray) else final_pos
    
    return result

def rotate_around_z(original_quat=None, original_pos=None, degrees=0):
    """
    计算四元数和/或3D位置点绕Z轴旋转指定角度后的新值
    
    参数:
        original_quat: 原始四元数 [w, x, y, z] (可选)
        original_pos: 原始位置 [x, y, z] (可选)
        degrees: 旋转角度（度数），正值为逆时针方向
        
    返回:
        字典包含:
        - 'new_quat': 旋转后的四元数 [w, x, y, z] (如果输入了original_quat)
        - 'new_pos': 旋转后的位置 [x, y, z] (如果输入了original_pos)
    """
    result = {}
    
    # 创建Z轴旋转
    z_rotation = Rotation.from_euler('z', degrees, degrees=True)
    
    # 处理四元数旋转
    if original_quat is not None:
        original_rot = Rotation.from_quat([original_quat[1], original_quat[2], original_quat[3], original_quat[0]])
        combined_rot = z_rotation * original_rot
        new_quat = combined_rot.as_quat()
        # result['new_quat'] = [new_quat[3], new_quat[0], new_quat[1], new_quat[2]]
        result['new_quat'] = [float(new_quat[3]), float(new_quat[0]), float(new_quat[1]), float(new_quat[2])]
    
    # 处理位置点旋转
    if original_pos is not None:
        # 将位置转换为齐次坐标并应用旋转
        rotated_pos = z_rotation.apply(original_pos)
        result['new_pos'] = rotated_pos.tolist() if isinstance(rotated_pos, np.ndarray) else rotated_pos
    
    return result

@register_problem
class Libero_Tabletop_Manipulation(BDDLBaseDomain):
    def __init__(self, bddl_file_name, *args, **kwargs):
        self.workspace_name = "main_table"
        self.visualization_sites_list = []
        if "table_full_size" in kwargs:
            self.table_full_size = table_full_size
        else:
            self.table_full_size = (1.0, 1.2, 0.05)
        self.table_offset = (0, 0, 0.90)
        # For z offset of environment fixtures
        self.z_offset = 0.01 - self.table_full_size[2]
        kwargs.update(
            {"robots": [f"Mounted{robot_name}" for robot_name in kwargs["robots"]]}
        )
        kwargs.update({"workspace_offset": self.table_offset})
        kwargs.update({"arena_type": "table"})

        if "scene_xml" not in kwargs or kwargs["scene_xml"] is None:
            kwargs.update({"scene_xml": "scenes/libero_tabletop_base_style.xml"})
        if "scene_properties" not in kwargs or kwargs["scene_properties"] is None:
            kwargs.update(
                {
                    "scene_properties": {
                        "floor_style": "light-gray",
                        "wall_style": "light-gray-plaster",
                    }
                }
            )

        super().__init__(bddl_file_name, *args, **kwargs)

    def _load_fixtures_in_arena(self, mujoco_arena):
        """Nothing extra to load in this simple problem."""
        for fixture_category in list(self.parsed_problem["fixtures"].keys()):
            if fixture_category == "table":
                continue

            for fixture_instance in self.parsed_problem["fixtures"][fixture_category]:
                self.fixtures_dict[fixture_instance] = get_object_fn(fixture_category)(
                    name=fixture_instance,
                    joints=None,
                )

    def _load_objects_in_arena(self, mujoco_arena):
        objects_dict = self.parsed_problem["objects"]
        for category_name in objects_dict.keys():
            for object_name in objects_dict[category_name]:
                self.objects_dict[object_name] = get_object_fn(category_name)(
                    name=object_name
                )

    def _load_sites_in_arena(self, mujoco_arena):
        # Create site objects
        object_sites_dict = {}
        region_dict = self.parsed_problem["regions"]
        for object_region_name in list(region_dict.keys()):

            if "main_table" in object_region_name:
                ranges = region_dict[object_region_name]["ranges"][0]
                assert ranges[2] >= ranges[0] and ranges[3] >= ranges[1]
                zone_size = ((ranges[2] - ranges[0]) / 2, (ranges[3] - ranges[1]) / 2)
                zone_centroid_xy = (
                    (ranges[2] + ranges[0]) / 2,
                    (ranges[3] + ranges[1]) / 2,
                )
                target_zone = TargetZone(
                    name=object_region_name,
                    rgba=region_dict[object_region_name]["rgba"],
                    zone_size=zone_size,
                    zone_centroid_xy=zone_centroid_xy,
                )
                object_sites_dict[object_region_name] = target_zone

                mujoco_arena.table_body.append(
                    new_site(
                        name=target_zone.name,
                        pos=target_zone.pos,
                        quat=target_zone.quat,
                        rgba=target_zone.rgba,
                        size=target_zone.size,
                        type="box",
                    )
                )
                continue
            # Otherwise the processing is consistent
            for query_dict in [self.objects_dict, self.fixtures_dict]:
                for (name, body) in query_dict.items():
                    try:
                        if "worldbody" not in list(body.__dict__.keys()):
                            # This is a special case for CompositeObject, we skip this as this is very rare in our benchmark
                            continue
                    except:
                        continue
                    for part in body.worldbody.find("body").findall(".//body"):
                        sites = part.findall(".//site")
                        joints = part.findall("./joint")
                        if sites == []:
                            break
                        for site in sites:
                            site_name = site.get("name")
                            if site_name == object_region_name:
                                object_sites_dict[object_region_name] = SiteObject(
                                    name=site_name,
                                    parent_name=body.name,
                                    joints=[joint.get("name") for joint in joints],
                                    size=site.get("size"),
                                    rgba=site.get("rgba"),
                                    site_type=site.get("type"),
                                    site_pos=site.get("pos"),
                                    site_quat=site.get("quat"),
                                    object_properties=body.object_properties,
                                )
        self.object_sites_dict = object_sites_dict

        # Keep track of visualization objects
        for query_dict in [self.fixtures_dict, self.objects_dict]:
            for name, body in query_dict.items():
                if body.object_properties["vis_site_names"] != {}:
                    self.visualization_sites_list.append(name)

    def _add_placement_initializer(self):
        """Very simple implementation at the moment. Will need to upgrade for other relations later."""
        super()._add_placement_initializer()

    def _check_success(self):
        """
        Check if the goal is achieved. Consider conjunction goals at the moment
        """
        goal_state = self.parsed_problem["goal_state"]
        result = True
        for state in goal_state:
            result = self._eval_predicate(state) and result
        return result

    def _eval_predicate(self, state):
        if len(state) == 3:
            # Checking binary logical predicates
            predicate_fn_name = state[0]
            object_1_name = state[1]
            object_2_name = state[2]
            return eval_predicate_fn(
                predicate_fn_name,
                self.object_states_dict[object_1_name],
                self.object_states_dict[object_2_name],
            )
        elif len(state) == 2:
            # Checking unary logical predicates
            predicate_fn_name = state[0]
            object_name = state[1]
            return eval_predicate_fn(
                predicate_fn_name, self.object_states_dict[object_name]
            )

    def _setup_references(self):
        super()._setup_references()

    def _post_process(self):
        super()._post_process()

        self.set_visualization()

    def set_visualization(self):

        for object_name in self.visualization_sites_list:
            for _, (site_name, site_visible) in (
                self.get_object(object_name).object_properties["vis_site_names"].items()
            ):
                vis_g_id = self.sim.model.site_name2id(site_name)
                if ((self.sim.model.site_rgba[vis_g_id][3] <= 0) and site_visible) or (
                    (self.sim.model.site_rgba[vis_g_id][3] > 0) and not site_visible
                ):
                    # We toggle the alpha value
                    self.sim.model.site_rgba[vis_g_id][3] = (
                        1 - self.sim.model.site_rgba[vis_g_id][3]
                    )

    def _setup_camera(self, mujoco_arena):
        pos_av = [0.6586131746834771, 0.0, 1.6103500240372423]
        quat_av = [
                0.6380177736282349,
                0.3048497438430786,
                0.30484986305236816,
                0.6380177736282349,
            ]
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=pos_av,
            quat=quat_av,
        )
        view_list = [30,60,90,120,180,240,270,300,330]
        for view in view_list:
            result_view = rotate_around_z(original_quat=quat_av, original_pos=pos_av, degrees=int(view))
            pos_view = [round(x,4) for x in result_view['new_pos']]
            quat_view = [round(x,4) for x in result_view['new_quat']]
            mujoco_arena.set_camera(
                camera_name=f"agentview_{str(view)}", pos=pos_view, quat=quat_view
            )

        up_list = [15, 30, 345]
        for up_view in up_list:
            result_up = rotate_around_y(original_quat=quat_av, original_pos=pos_av, degrees=int(up_view))
            pos_up = [round(x,4) for x in result_up['new_pos']]
            quat_up = [round(x,4) for x in result_up['new_quat']]
            mujoco_arena.set_camera(
                camera_name=f"agentview_up_{str(up_view)}", pos=pos_up, quat=quat_up
            )
            for view in view_list:
                result_view = rotate_around_z(original_quat=quat_up, original_pos=pos_up, degrees=int(view))
                pos_view = [round(x,4) for x in result_view['new_pos']]
                quat_view = [round(x,4) for x in result_view['new_quat']]
                mujoco_arena.set_camera(
                    camera_name=f"agentview_up_{str(up_view)}_{str(view)}", pos=pos_view, quat=quat_view
                )

        # For visualization purpose
        mujoco_arena.set_camera(
            camera_name="frontview", pos=[1.0, 0.0, 1.48], quat=[0.56, 0.43, 0.43, 0.56]
        )
        mujoco_arena.set_camera(
            camera_name="galleryview",
            pos=[2.844547668904445, 2.1279684793440667, 3.128616846013882],
            quat=[
                0.42261379957199097,
                0.23374411463737488,
                0.41646939516067505,
                0.7702690958976746,
            ],
        )
