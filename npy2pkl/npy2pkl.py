import numpy as np
import torch
import pickle
import os
from pytorch3d.transforms import (matrix_to_axis_angle, rotation_6d_to_matrix)

def ax_from_6v(q):
    assert q.shape[-1] == 6
    mat = rotation_6d_to_matrix(q)
    ax = matrix_to_axis_angle(mat)
    # ax = matrix_to_quaternion(mat)
    # ax = quaternion_to_axis_angle(ax)
    return ax
def convert_npy_to_pkl(npy_path, pkl_path):
    """
    将139或319维npy数据转换为标准SMPL格式的pkl文件
    参数:
        npy_path: 输入npy文件路径
        pkl_path: 输出pkl文件路径
    """
    # 加载原始数据
    raw_data = np.load(npy_path)  # [帧数, 139]

    # 检查数据维度并处理319维情况
    if raw_data.shape[1] == 319:
        print(f"检测到319维数据，自动裁剪为139维: {npy_path}")
        # 319维结构: [4自定义 + 3平移 + 22身体关节×6D + 30手部关节×6D]
        # 我们只需要前139维: [4 + 3 + 22×6 = 139]
        raw_data = raw_data[:, :139]
    elif raw_data.shape[1] != 139:
        raise ValueError(f"不支持的输入维度: {raw_data.shape[1]}，应为139或319")

    # 分解数据段
    # custom = raw_data[:, :4]  # 自定义数据（占位符）
    root_trans = raw_data[:, 4:7]  # 根节点平移 [N,3]
    body_6d = raw_data[:, 7:]  # SMPL身体姿态 [N,132]

    # 6D转轴角格式
    # body_6d_tensor = torch.tensor(body_6d.reshape(-1, 22, 6))  # [N,22,6]
    # rotation_matrix = rotation_6d_to_matrix(body_6d_tensor)
    # body_axis_angle = matrix_to_axis_angle(rotation_matrix)  # [N,22,3]

    body_6d_tensor = torch.from_numpy(body_6d).reshape(-1, 22, 6)  # [N,22,6]
    body_axis_angle = ax_from_6v(body_6d_tensor)  # [N,22,3]

    # 补全至24关节
    # zeros_hands = torch.zeros((body_axis_angle.shape[0], 2, 3),
    #                          dtype=body_axis_angle.dtype,
    #                          device=body_axis_angle.device)  # [N,2,3]

    # 补全至SMPLH格式（使用NumPy操作）
    zeros_hands = np.zeros((body_axis_angle.shape[0], 30, 3), dtype=np.float32)
    full_poses = np.concatenate([body_axis_angle, zeros_hands], axis=1)  # [N,52,3]
    full_poses = full_poses.reshape(-1, 156)  # 转换为156维

    # 保存为纯NumPy数据
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            "smpl_poses": full_poses.astype(np.float32),
            "smpl_trans": root_trans.astype(np.float32)
        }, f)


def batch_convert_npy_to_pkl(input_dir, output_dir):
    """
    批量转换npy文件为pkl文件
    参数:
        input_dir: 输入npy文件夹路径
        output_dir: 输出pkl文件夹路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录中的所有npy文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.npy'):
            # 构造完整路径
            npy_path = os.path.join(input_dir, filename)
            pkl_filename = os.path.splitext(filename)[0] + '.pkl'
            pkl_path = os.path.join(output_dir, pkl_filename)

            # 转换文件
            print(f"正在处理: {filename} -> {pkl_filename}")
            try:
                convert_npy_to_pkl(npy_path, pkl_path)
                print(f"成功转换: {filename}")
            except Exception as e:
                print(f"转换失败 {filename}: {str(e)}")


if __name__ == "__main__":
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 设置输入输出目录
    input_dir = os.path.join(script_dir, 'dance_npy')
    output_dir = os.path.join(script_dir, 'dance_pkl')

    # 执行批量转换
    print(f"开始批量转换，从 {input_dir} 到 {output_dir}")
    batch_convert_npy_to_pkl(input_dir, output_dir)
    print("所有文件处理完成！")