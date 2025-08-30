from libero.libero.envs import OffScreenRenderEnv
from PIL import Image

bddl_file_path = "/home/ps/LIBERO/notebooks/custom_pddl/KITCHEN_DEMO_SCENE_libero_demo_behaviors.bddl"
save_image_path = "/home/ps/LIBERO/notebooks/custom_pddl/image.jpg"

with open(bddl_file_path, "r") as f:
    print(f.read())

env_args = {
    "bddl_file_name": bddl_file_path,
    "camera_heights": 256,
    "camera_widths": 256
}

env = OffScreenRenderEnv(**env_args)
obs = env.reset()
image = Image.fromarray(obs["agentview_image"][::-1])
image.save(save_image_path)
env.close()
