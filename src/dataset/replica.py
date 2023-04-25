import os.path as osp

import cv2
import numpy as np
import torch
import csv
from glob import glob
from torch.utils.data import Dataset


class DataLoader(Dataset):
    def __init__(self, data_path, use_gt=False, max_depth=-1) -> None:
        self.data_path = data_path
        self.num_imgs = len(glob(osp.join(self.data_path, "results/*.jpg")))
        self.max_depth = max_depth
        self.use_gt = use_gt
        self.K = self.load_intrinsic()
        self.gt_pose = self.load_gt_pose()

    def load_intrinsic(self):
        K = np.eye(3)
        K[0, 0] = K[1, 1] = 600
        K[0, 2] = 599.5
        K[1, 2] = 339.5

        return K

    def get_init_pose(self):
        if self.gt_pose is not None:
            return self.gt_pose[0].reshape(4, 4)
        else:
            return np.eye(4)

    def load_gt_pose(self):
        gt_file = osp.join(self.data_path, 'traj.txt')
        gt_pose = np.loadtxt(gt_file)
        return gt_pose

    def load_depth(self, index):
        depth = cv2.imread(
            osp.join(self.data_path, 'results/depth{:06d}.png'.format(index)), -1)
        depth = depth / 6553.5
        if self.max_depth > 0:
            depth[depth > self.max_depth] = 0
        return depth

    # add noise in depth
    def add_depth_noise(self, depth):
        # means = 0.1125 * (depth_imgs[i]* far)**2 + 4.8875
        # std = 2.925 * (depth_imgs[i]* far)**2 + 3.325
        # bias = torch.normal(mean = means,std = std) / far / 1000
        # bias = bias.to(device)
        # depth_imgs[i] = depth_imgs[i] + bias
        means = 0.1125 * (depth * self.max_depth)**2 + 4.8875
        std = 2.925 * (depth * self.max_depth)**2 + 3.325
        bias = np.random.normal(means, std) / self.max_depth / 1000
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # bias = bias.to(device)
        depth = depth + bias
        return depth

    def load_image(self, index):
        rgb = cv2.imread(
            osp.join(self.data_path, 'results/frame{:06d}.jpg'.format(index)), -1)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return rgb / 255.0

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, index):
        img = torch.from_numpy(self.load_image(index)).float()
        depth = self.load_depth(index)
        depth = self.add_depth_noise(depth)
        depth = None if depth is None else torch.from_numpy(depth).float()
        pose = self.gt_pose[index] if self.use_gt else None
        return index, img, depth, self.K, pose


if __name__ == '__main__':
    import sys
    # loader = DataLoader(sys.argv[1])
    loader = DataLoader("/remote-home/ums_sunjiahao/datasets/Replica/room0")    # load replica room0
    for data in loader:
        index, img, depth, K, _ = data
        print(K)
        print(index, img.shape)
        print(depth.numpy())
        print(depth.shape)
        cv2.imshow('img', img.numpy())
        cv2.imshow('depth', depth.numpy())
        cv2.waitKey(1)
