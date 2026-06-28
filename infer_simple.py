# -*- coding: utf-8 -*-
"""
MedSegX 简单推理脚本。
输入：图像目录 + 权重文件 + task_name，输出所有图像的掩码（PNG）。
"""

import argparse
import os
join = os.path.join

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Resize
from tqdm import tqdm

from segment_anything import sam_model_registry, sam_model_checkpoint
from segment_anything.utils.transforms import ResizeLongestSide
from model import MedSAM, MedSegX
from data.datainfo import (
    modal_dict,
    modal_map,
    organ_level_1_dict,
    organ_level_1_map,
    organ_level_2_dict,
    organ_level_2_map,
    organ_level_3_dict,
    organ_level_3_map,
    task_idx,
)

IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def parse_task(task_name: str):
    """从 task_name（如 'US_ThyroidNodule'）解析 modal 和 organ 索引"""
    modal = task_name.split('_')[0]
    organ = ('').join(task_name.split('_')[1:]).rstrip('0123456789')

    modal_idx = modal_map[modal_dict[modal]]
    organ_level_1 = next(organ_level_1_map[k] for k, v in organ_level_1_dict.items() if organ in v)
    organ_level_2 = next(organ_level_2_map[k] for k, v in organ_level_2_dict.items() if organ in v)
    organ_level_3 = next(organ_level_3_map[k] for k, v in organ_level_3_dict.items() if organ in v)
    organ_level_4 = task_idx[organ]
    return modal_idx, (organ_level_1, organ_level_2, organ_level_3, organ_level_4)


def load_image(path: str) -> np.ndarray:
    """加载任意图像为 HWC float32 RGB numpy 数组"""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32)


@torch.no_grad()
def infer_single(model, image_np: np.ndarray, modal: int, organ: tuple,
                 img_size: int, device: torch.device):
    """
    对单张图像推理，返回 (H, W) uint8 二值掩码。
    image_np: HWC float32 RGB
    """
    model.eval()
    h_orig, w_orig = image_np.shape[:2]

    # --- 图像转 tensor ---
    img_tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # C,H,W

    # --- 全图 box prompt ---
    box = torch.tensor([[0, 0, w_orig, h_orig]], dtype=torch.float32)

    # --- Resize（如果尺寸不是 img_size）---
    box_transform = ResizeLongestSide(img_size)
    box = box_transform.apply_boxes_torch(
        box.reshape(-1, 2, 2), (h_orig, w_orig)
    ).reshape(-1, 4)

    img_resize = Resize((img_size, img_size), antialias=True)
    img_tensor = img_resize(img_tensor.unsqueeze(0))  # (1, 3, img_size, img_size)

    # --- 移到设备 ---
    img_tensor = img_tensor.to(device)
    box = box.to(device)

    # --- SAM preprocess ---
    img_tensor = model.sam.preprocess(img_tensor)

    # --- Prompt encoder ---
    sparse_emb, dense_emb = model.sam.prompt_encoder(
        points=None, boxes=box[:, None, :], masks=None
    )

    # --- Modal & Organ embedding ---
    batch_size = img_tensor.shape[0]
    modal_index = model.sam.image_encoder.modal_index[modal]
    modal_embed = model.sam.image_encoder.modal_embed(modal_index)

    organ_1, organ_2, organ_3, organ_4 = organ
    organ_index_0 = torch.zeros(batch_size, dtype=torch.long, device=device)
    organ_index_4 = torch.tensor([organ_4], dtype=torch.long, device=device)
    organ_embed = (
        model.sam.image_encoder.organ_embed[0](organ_index_0),
        model.sam.image_encoder.organ_embed[1](
            model.sam.image_encoder.organ_index_1[organ_1]),
        model.sam.image_encoder.organ_embed[2](
            model.sam.image_encoder.organ_index_2[organ_2]),
        model.sam.image_encoder.organ_embed[3](
            model.sam.image_encoder.organ_index_3[organ_3]),
        model.sam.image_encoder.organ_embed[4](organ_index_4),
    )

    # --- Image encoder ---
    image_embedding, _ = model.sam.image_encoder(img_tensor, modal_embed, organ_embed)

    # --- Mask decoder ---
    mask_pred, iou_pred = model.sam.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=True,
    )  # mask_pred: (1, 3, 256, 256), iou_pred: (1, 3)

    # --- 选最佳 mask ---
    best_idx = iou_pred.argmax(dim=1)
    mask_prob = torch.sigmoid(mask_pred)
    chosen = mask_prob[0, best_idx[0]]  # (256, 256)
    chosen = (chosen > 0.5).to(torch.uint8)

    # --- Resize 回原图尺寸 ---
    chosen = chosen.unsqueeze(0).unsqueeze(0).float()
    chosen = F.interpolate(chosen, size=(h_orig, w_orig), mode="bilinear", antialias=True)
    mask_np = (chosen.squeeze() > 0.5).to(torch.uint8).cpu().numpy()

    return mask_np


def main():
    parser = argparse.ArgumentParser("MedSegX Simple Inference")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="输入图像目录（支持 PNG/JPG/BMP/TIFF 等）")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出掩码保存目录")
    parser.add_argument("--task_name", type=str, required=True,
                        help="任务名，如 US_ThyroidNodule")
    parser.add_argument("--checkpoint", type=str, default="./playground/SAM",
                        help="SAM checkpoint 目录")
    parser.add_argument("--model_type", type=str, default="vit_b",
                        help="SAM 模型规模：vit_b / vit_l / vit_h")
    parser.add_argument("--model_weight", type=str, required=True,
                        help="MedSegX 权重文件路径")
    parser.add_argument("--method", type=str, default="medsegx",
                        help="模型方法：medsegx / medsam")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # --- 收集图像文件 ---
    img_files = sorted([
        join(args.input_dir, f) for f in os.listdir(args.input_dir)
        if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
    ])
    if len(img_files) == 0:
        raise RuntimeError(f"在 {args.input_dir} 中未找到图像文件（支持：{IMG_EXTENSIONS}）")

    print(f"找到 {len(img_files)} 张图像")

    # --- 解析 task ---
    modal, organ = parse_task(args.task_name)
    print(f"任务: {args.task_name}, modal={modal}, organ={organ}")

    # --- 加载模型 ---
    device = torch.device(args.device)
    sam_ckpt = join(args.checkpoint, sam_model_checkpoint[args.model_type])
    sam_model = sam_model_registry[args.model_type](
        image_size=256, keep_resolution=True, checkpoint=sam_ckpt
    )
    if args.method == "medsegx":
        model = MedSegX(sam_model, bottleneck_dim=16, embedding_dim=16, expert_num=4).to(device)
    elif args.method == "medsam":
        model = MedSAM(sam_model).to(device)
    else:
        raise NotImplementedError(f"不支持的方法: {args.method}")

    ckpt = torch.load(args.model_weight, map_location=device)
    model.load_parameters(ckpt["model"])
    print(f"已加载模型权重: {args.model_weight}")

    img_size = model.sam.image_encoder.img_size  # 256

    # --- 逐张推理 ---
    os.makedirs(args.output_dir, exist_ok=True)
    for img_path in tqdm(img_files, desc="推理中"):
        image_np = load_image(img_path)
        mask = infer_single(model, image_np, modal, organ, img_size, device)

        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = join(args.output_dir, f"{base}_mask.png")
        Image.fromarray(mask * 255).save(out_path)

    print(f"完成！掩码已保存至 {args.output_dir}")


if __name__ == "__main__":
    main()
