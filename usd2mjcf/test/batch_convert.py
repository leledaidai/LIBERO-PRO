import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert USD to MJCF")
    parser.add_argument("input_path", type=str, help="Path to input USD file")
    parser.add_argument("--generate_collision", action='store_true', help="Generate collision meshes for the MJCF model")
    parser.add_argument("--preprocess_resolution", type=int, default=20, help="Preprocessing voxelization resolution for convex decomposition")
    parser.add_argument("--resolution", type=int, default=2000, help="Main voxelization resolution for convex decomposition")
    
    args = parser.parse_args()
    input_path = args.input_path
    generate_collision = args.generate_collision
    preprocess_resolution = args.preprocess_resolution
    resolution = args.resolution
    
    tmp = " --generate_collision --preprocess_resolution={preprocess_resolution} --resolution={resolution}" if generate_collision else ""
    os.system("echo 'Hello World'")
    if os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for file in files:
                if ".tmp.usd" in file:
                    continue
                if file.endswith(".usd"):
                    usd_path = os.path.join(root, file)
                    exec = "python test/usd2mjcf_test.py " + usd_path + tmp
                    # print(exec)
                    os.system(exec)
    else:
        exec = "python test/usd2mjcf_test.py " + input_path + tmp
        # print(exec)
        os.system(exec)