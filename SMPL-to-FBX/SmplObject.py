import glob
import os
import pickle
from typing import Dict, Tuple

import numpy as np

def smplh2smpl_pose(poses):
    "convert SMPLH pose to SMPL pose(156, ) -> (72, )"
    assert len(poses) == 156, f'the given pose shape is not correct: {poses.shape}'
    smpl_pose = np.concatenate([poses[:69], poses[111:114]])  # Extract relevant parts for SMPL
    return smpl_pose
class SmplObjects(object):
    joints = [
        "m_avg_Pelvis",
        "m_avg_L_Hip",
        "m_avg_R_Hip",
        "m_avg_Spine1",
        "m_avg_L_Knee",
        "m_avg_R_Knee",
        "m_avg_Spine2",
        "m_avg_L_Ankle",
        "m_avg_R_Ankle",
        "m_avg_Spine3",
        "m_avg_L_Foot",
        "m_avg_R_Foot",
        "m_avg_Neck",
        "m_avg_L_Collar",
        "m_avg_R_Collar",
        "m_avg_Head",
        "m_avg_L_Shoulder",
        "m_avg_R_Shoulder",
        "m_avg_L_Elbow",
        "m_avg_R_Elbow",
        "m_avg_L_Wrist",
        "m_avg_R_Wrist",
        "m_avg_L_Hand",
        "m_avg_R_Hand",
    ]

    def __init__(self, read_path):

    #     self.files = {}
    #
    #     if os.path.isfile(read_path):
    #         paths = [read_path]
    #     else:
    #         paths = sorted(glob.glob(os.path.join(read_path, "*.pkl")))
    #
    #     for path in paths:
    #         filename = path.split("/")[-1]
    #         with open(path, "rb") as fp:
    #             data = pickle.load(fp)
    #         self.files[filename] = {
    #             "smpl_poses": data["smpl_poses"],
    #             "smpl_trans": data["smpl_trans"],
    #         }
    #     self.keys = [key for key in self.files.keys()]
    #
    # def __len__(self):
    #     return len(self.keys)
    #
    # def __getitem__(self, idx: int) -> Tuple[str, Dict]:
    #     key = self.keys[idx]
    #     return key, self.files[key]

        self.files = {}

        if os.path.isfile(read_path):
            paths = [read_path]
        else:
            paths = sorted(glob.glob(os.path.join(read_path, "*.pkl")))

        for path in paths:
            filename = os.path.basename(path)
            with open(path, "rb") as fp:
                data = pickle.load(fp)

            # 原数据 shape: (N, 156)
            poses = data["smpl_poses"]  # SMPL-H 全 52 关节的旋转
            trans = data["smpl_trans"]

            # 只保留前 24 个关节，对应 72 维
            # 如果 poses 是 (N, 156)，这里取 poses[:, :72]
            # 就能截断掉多余的手指关节数据
            # poses_24 = poses[:, :72]
            poses_72 = np.concatenate([poses[:, :69], poses[:, 111:114]], axis=-1)
            # print(poses_72)

            self.files[filename] = {
                "smpl_poses": poses_72,
                "smpl_trans": trans,
            }

        self.keys = list(self.files.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int) -> Tuple[str, Dict]:
        key = self.keys[idx]
        return key, self.files[key]
