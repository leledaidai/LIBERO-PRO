export PYTHONPATH="/home/ps/LIBERO/usd2mjcf/"
python decrypt_usd.py /home/ps/BEHAVIOR-1K/OmniGibson/omnigibson/data/og_dataset/objects/bottle_of_gin/qzgcdx/usd/try.usd /home/ps/BEHAVIOR-1K/OmniGibson/omnigibson/data/og_dataset/objects/bottle_of_gin/qzgcdx/usd/
python3 test/usd2mjcf_test.py /home/ps/BEHAVIOR-1K/OmniGibson/omnigibson/data/og_dataset/objects/bottle_of_gin/qzgcdx/usd/try.usd --generate_collision

