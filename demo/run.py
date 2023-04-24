import os  # noqa
import sys  # noqa
sys.path.insert(0, os.path.abspath('/remote-home/ums_sunjiahao/Vox-Fusion/src')) # noqa
# sys.path.insert(0, os.path.abspath('/remote-home/ums_sunjiahao/Vox-Fusion/configs')) # config path
import random
from parser import get_parser
import numpy as np
import torch
from voxslam import VoxSLAM

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    # args = get_parser().parse_args()
    args = get_parser().parse_args(["/remote-home/ums_sunjiahao/Vox-Fusion/configs/replica/room_0.yaml"])
    if hasattr(args, 'seeding'):
        setup_seed(args.seeding)

    slam = VoxSLAM(args)
    slam.start()
    slam.wait_child_processes()