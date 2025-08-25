import torch

init_states_path = "/inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/LIBERO/libero/libero/init_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.pruned_init"
init_states = torch.load(init_states_path)
print(f"init_states.shape: {init_states.shape}")

print(init_states[0])

# robot_init_state = init_states[:7]
# print(f"robot_init_state.shape: {robot_init_state.shape}")
