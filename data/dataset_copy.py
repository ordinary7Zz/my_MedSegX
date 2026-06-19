"""
=============================================================================
数据集加载模块 (dataset_copy.py)
=============================================================================

本模块定义了两个 PyTorch Dataset 类，用于加载医学图像分割数据：

1. GeneralMedSegDB：通用医学分割数据集，遍历所有数据集和任务加载数据。
2. TaskMedSegDB：单任务医学分割数据集，针对某个特定任务/数据集加载数据。

【数据组织结构】
  - npy_imgs/：存放图像的 .npy 文件（H x W x 3，RGB 格式）
  - npy_gts/：存放 ground truth mask 的 .npy 文件（H x W，二值前景）

【运行流程】
  1. __init__：扫描 npy_gts 目录，收集所有 ground truth 文件路径，
     并根据目录层级解析出任务名称（格式为 "模态_器官"）。
  2. __getitem__：
     a) 加载图像 (.npy) 和对应的 ground truth mask (.npy)
     b) 从 GT mask 中提取前景区域的最小外接矩形 (bounding box)，
        格式为 [x_min, y_min, x_max, y_max]
     c) 对 bounding box 进行随机扰动（扩大）：
        - 训练时 (train=True)：每边随机扩大 0~20 像素（数据增强）
        - 推理时 (train=False)：每边也随机扩大 0~20 像素（模拟真实场景中
          用户框选不精确的情况）
     d) 根据任务名称解析模态 (modal) 和器官 (organ) 的层级索引，
        包括：body parts → body subregions → organs/tissues → tasks
     e) 返回包含图像、box prompt、模态/器官信息的字典，以及 GT mask 标签

【Box Prompt 的作用】
  Bounding box 作为 SAM 模型的 prompt 输入，告知模型需要分割的目标大致位置。
  模型通过 prompt encoder 将 box 编码为 sparse embedding，引导 mask decoder
  在该区域内生成精确的分割结果。
=============================================================================
"""

import os
join = os.path.join

import torch
from torch.utils.data import Dataset
import numpy as np

from .datainfo import *


class GeneralMedSegDB(Dataset):
    def __init__(self, data_root, train=True):
        self.data_root = data_root
        self.train = train

        self.file_names = []
        self.task_names = []

        # iterate over datasets
        for dataset in sorted(os.listdir(data_root)):
            dataset_dir = join(data_root, dataset)

            # task names in "modal_organ" format
            for task in sorted(os.listdir(dataset_dir)):
                task_dir = join(dataset_dir, task)

                # modality without multiple sequences
                if "npy_gts" in os.listdir(task_dir):
                    files = sorted(os.listdir(join(task_dir, "npy_gts")))
                    self.file_names += [join(task_dir, "npy_gts", f) for f in files]
                    self.task_names += [task] * len(files)

                # modality with multiple sequences
                else:
                    for sequence in os.listdir(task_dir):
                        sequence_dir = join(task_dir, sequence)
                        files = sorted(os.listdir(join(sequence_dir, "npy_gts")))
                        self.file_names += [join(sequence_dir, "npy_gts", f) for f in files]
                        self.task_names += [task] * len(files)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        img = np.load(self.file_names[index].replace("npy_gts", "npy_imgs")).transpose(2, 0, 1)
        gt = np.load(self.file_names[index])

        # get bounding box coordinates from gt mask
        y_indices, x_indices = np.where(gt > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # add perturbation to bounding box coordinates
        if self.train:
            H, W = gt.shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
        box = np.array([x_min, y_min, x_max, y_max])

        # task with modal_organ format
        task = self.task_names[index]
        modal = task.split('_')[0]
        organ = ('').join(task.split('_')[1:])

        # map modal and organ to index
        modal = modal_map[modal_dict[modal]]
        # remove spine index
        organ = organ.rstrip('0123456789')
        # body parts
        for k, v in organ_level_1_dict.items():
            if organ in v:
                organ_level_1 = organ_level_1_map[k]
                break
        # body subregions
        for k, v in organ_level_2_dict.items():
            if organ in v:
                organ_level_2 = organ_level_2_map[k]
                break
        # organs and tissues
        for k, v in organ_level_3_dict.items():
            if organ in v:
                organ_level_3 = organ_level_3_map[k]
                break
        # tasks
        organ_level_4 = task_idx[organ]

        data = {
            "img": torch.tensor(img).float(),
            "box": torch.tensor(box).float(),
            "modal": modal,
            "organ": (organ_level_1, organ_level_2, organ_level_3, organ_level_4),
            "name": self.file_names[index],
        }
        return data, torch.tensor(gt[None, :, :]).long()


class TaskMedSegDB(Dataset):
    def __init__(self, data_root, train=True, sequence=False):
        self.data_root = data_root
        self.train = train

        files = sorted(os.listdir(join(data_root, "npy_gts")))
        self.file_names = [join(data_root, "npy_gts", f) for f in files]

        # ID task dataset with the dataset name before the task name
        if "eval/ID" in data_root:
            # modality without multiple sequences
            if not sequence:
                self.task = data_root.split('/')[-1]
            # modality with multiple sequences
            else:
                self.task = data_root.split('/')[-2]
        # OOD task dataset with the dataset name after the task name
        else:
            # modality without multiple sequences
            if not sequence:
                self.task = data_root.split('/')[-3]
            # modality with multiple sequences
            else:
                self.task = data_root.split('/')[-4]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        img = np.load(self.file_names[index].replace("npy_gts", "npy_imgs")).transpose(2, 0, 1)
        gt = np.load(self.file_names[index])

        # get bounding box coordinates from gt mask
        y_indices, x_indices = np.where(gt > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # add perturbation to bounding box coordinates
        H, W = gt.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        box = np.array([x_min, y_min, x_max, y_max])

        # task with modal_organ format
        task = self.task
        modal = task.split('_')[0]
        organ = ('').join(task.split('_')[1:])

        # map modal and organ to index
        modal = modal_map[modal_dict[modal]]
        # remove spine index
        organ = organ.rstrip('0123456789')
        # body parts
        for k, v in organ_level_1_dict.items():
            if organ in v:
                organ_level_1 = organ_level_1_map[k]
                break
        # body subregions
        for k, v in organ_level_2_dict.items():
            if organ in v:
                organ_level_2 = organ_level_2_map[k]
                break
        # organs and tissues
        for k, v in organ_level_3_dict.items():
            if organ in v:
                organ_level_3 = organ_level_3_map[k]
                break
        # tasks
        organ_level_4 = task_idx[organ]

        data = {
            "img": torch.tensor(img).float(),
            "box": torch.tensor(box).float(),
            "modal": modal,
            "organ": (organ_level_1, organ_level_2, organ_level_3, organ_level_4),
            "name": self.file_names[index],
        }
        return data, torch.tensor(gt[None, :, :]).long()
