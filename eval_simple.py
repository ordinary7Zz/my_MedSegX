# -*- coding: utf-8 -*-
"""
MedSegX 简单评估脚本。
输入：图像目录 + GT mask 目录 + 微调权重，计算 DSC 指标。
支持 PNG/JPG 等通用格式，图像和 mask 按文件名匹配。
"""

import argparse
import os
join = os.path.join

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt as edt
from torchvision.transforms import Resize
from tqdm import tqdm

from segment_anything import sam_model_registry, sam_model_checkpoint
from segment_anything.utils.transforms import ResizeLongestSide
from model import MedSAM, MedSegX
from data.datainfo import (
    modal_dict, modal_map,
    organ_level_1_dict, organ_level_1_map,
    organ_level_2_dict, organ_level_2_map,
    organ_level_3_dict, organ_level_3_map,
    task_idx,
)

IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def parse_task(task_name: str):
    modal = task_name.split('_')[0]
    organ = ('').join(task_name.split('_')[1:]).rstrip('0123456789')
    modal_idx = modal_map[modal_dict[modal]]
    l1 = next(organ_level_1_map[k] for k, v in organ_level_1_dict.items() if organ in v)
    l2 = next(organ_level_2_map[k] for k, v in organ_level_2_dict.items() if organ in v)
    l3 = next(organ_level_3_map[k] for k, v in organ_level_3_dict.items() if organ in v)
    l4 = task_idx[organ]
    return modal_idx, (l1, l2, l3, l4)


def load_as_float32(path: str) -> np.ndarray:
    """加载图像为 HWC float32 RGB"""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32)


def load_mask(path: str, h: int, w: int) -> np.ndarray:
    """加载 GT mask，转二值并 resize 到 (h, w)"""
    mask = Image.open(path).convert("L")
    mask = mask.resize((w, h), Image.NEAREST)
    mask = np.array(mask, dtype=np.float32) / 255.0
    return (mask > 0.5).astype(np.float32)


def dice_coeff(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """计算单个样本的 Dice coefficient"""
    pred = pred.flatten().float()
    gt = gt.flatten().float()
    intersection = (pred * gt).sum()
    return (2. * intersection / (pred.sum() + gt.sum() + 1e-8)).item()


def _hd95_one_sided(x: np.ndarray, y: np.ndarray) -> float:
    """x 中每个前景点到 y 最近前景点的距离的 95 百分位。

    edt(~y) 得到每个像素到 y 前景最近点的欧氏距离图；
    取 x 前景处的距离值，求 95 百分位。
    """
    distances = edt(~y)
    indexes = np.nonzero(x)
    return float(np.percentile(distances[indexes], 95))


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """计算 Hausdorff Distance 95th percentile（像素单位）。

    基于 scipy 距离变换实现，不依赖 monai。
    边界情况约定（与 medsegx_infer/utils/metrics.py 对齐）：
    - pred 与 gt 均非空 → 正常计算对称 HD95 = max(hd(pred→gt), hd(gt→pred))；
    - pred 非空、gt 为空（假阳性）→ 0.0；
    - pred 为空、gt 非空（漏检/假阴性）→ 0.0；
    - pred 与 gt 均为空（真阴性）→ 0.0。
    """
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)

    if not pred_b.any() or not gt_b.any():
        return 0.0

    hd1 = _hd95_one_sided(pred_b, gt_b)
    hd2 = _hd95_one_sided(gt_b, pred_b)
    return float(max(hd1, hd2))


@torch.no_grad()
def infer_single(model, image_np: np.ndarray, modal: int, organ: tuple,
                 img_size: int, device: torch.device):
    """返回 (H, W) uint8 二值掩码"""
    model.eval()
    h_orig, w_orig = image_np.shape[:2]

    img_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
    box = torch.tensor([[0, 0, w_orig, h_orig]], dtype=torch.float32)

    box_transform = ResizeLongestSide(img_size)
    box = box_transform.apply_boxes_torch(
        box.reshape(-1, 2, 2), (h_orig, w_orig)).reshape(-1, 4)

    img_resize = Resize((img_size, img_size), antialias=True)
    img_tensor = img_resize(img_tensor.unsqueeze(0))

    img_tensor = img_tensor.to(device)
    box = box.to(device)
    img_tensor = model.sam.preprocess(img_tensor)

    sparse_emb, dense_emb = model.sam.prompt_encoder(
        points=None, boxes=box[:, None, :], masks=None)

    batch_size = img_tensor.shape[0]
    modal = torch.tensor([modal], dtype=torch.long, device=device)
    modal_index = model.sam.image_encoder.modal_index[modal]
    modal_embed = model.sam.image_encoder.modal_embed(modal_index)

    o1, o2, o3, o4 = organ
    o1, o2, o3, o4 = [torch.tensor([x], dtype=torch.long, device=device) for x in (o1, o2, o3, o4)]
    organ_index_0 = torch.zeros(batch_size, dtype=torch.long, device=device)
    organ_embed = (
        model.sam.image_encoder.organ_embed[0](organ_index_0),
        model.sam.image_encoder.organ_embed[1](model.sam.image_encoder.organ_index_1[o1]),
        model.sam.image_encoder.organ_embed[2](model.sam.image_encoder.organ_index_2[o2]),
        model.sam.image_encoder.organ_embed[3](model.sam.image_encoder.organ_index_3[o3]),
        model.sam.image_encoder.organ_embed[4](o4),
    )

    image_embedding, _ = model.sam.image_encoder(img_tensor, modal_embed, organ_embed)
    mask_pred, iou_pred = model.sam.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=True,
    )

    best_idx = iou_pred.argmax(dim=1)
    mask_prob = torch.sigmoid(mask_pred)
    chosen = mask_prob[0, best_idx[0]]
    chosen = (chosen > 0.5).to(torch.uint8)

    chosen = chosen.unsqueeze(0).unsqueeze(0).float()
    chosen = F.interpolate(chosen, size=(h_orig, w_orig), mode="bilinear", antialias=True)
    return (chosen.squeeze() > 0.5).to(torch.uint8).cpu().numpy()


def main():
    parser = argparse.ArgumentParser("MedSegX Simple Evaluation")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="测试图像目录（PNG/JPG 等）")
    parser.add_argument("--mask_dir", type=str, required=True,
                        help="GT mask 目录，文件名需与图像一一对应")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="结果保存目录")
    parser.add_argument("--task_name", type=str, required=True,
                        help="任务名，如 US_GlndThyroid")
    parser.add_argument("--checkpoint", type=str, default="./playground/SAM")
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--model_weight", type=str, required=True,
                        help="微调后的 MedSegX 权重")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_masks", action="store_true", default=False,
                        help="是否保存预测 mask（默认不保存）")
    args = parser.parse_args()

    # 收集图像文件，按文件名排序匹配
    img_files = sorted([
        f for f in os.listdir(args.image_dir)
        if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
    ])
    if not img_files:
        raise RuntimeError(f"未找到图像文件: {args.image_dir}")

    print(f"找到 {len(img_files)} 张测试图像")

    # 解析 task
    modal, organ = parse_task(args.task_name)
    print(f"任务: {args.task_name}, modal={modal}, organ={organ}")

    # 加载模型
    device = torch.device(args.device)
    sam_ckpt = join(args.checkpoint, sam_model_checkpoint[args.model_type])
    sam_model = sam_model_registry[args.model_type](
        image_size=256, keep_resolution=True, checkpoint=sam_ckpt)
    model = MedSegX(sam_model, bottleneck_dim=16, embedding_dim=16, expert_num=4).to(device)
    ckpt = torch.load(args.model_weight, map_location=device)
    model.load_parameters(ckpt["model"])
    print(f"已加载权重: {args.model_weight}")

    img_size = model.sam.image_encoder.img_size
    os.makedirs(args.output_dir, exist_ok=True)

    dsc_list = []
    hd95_list = []
    results = []

    for fname in tqdm(img_files, desc="评估中"):
        img_path = join(args.image_dir, fname)
        image_np = load_as_float32(img_path)
        h, w = image_np.shape[:2]

        pred_mask = infer_single(model, image_np, modal, organ, img_size, device)

        # 加载 GT（自动匹配同名文件）
        mask_name = os.path.splitext(fname)[0]
        # 尝试多种 mask 后缀
        gt_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
            candidate = join(args.mask_dir, mask_name + ext)
            if os.path.exists(candidate):
                gt_path = candidate
                break
        if gt_path is None:
            print(f"[警告] 未找到 {fname} 对应的 GT mask，跳过")
            continue

        gt_mask = load_mask(gt_path, h, w)

        dsc = dice_coeff(torch.from_numpy(pred_mask), torch.from_numpy(gt_mask))
        hd = hd95(pred_mask, gt_mask)
        dsc_list.append(dsc)
        hd95_list.append(hd)
        results.append((fname, dsc, hd))

        # 可选保存预测 mask
        if args.save_masks:
            base = os.path.splitext(fname)[0]
            out_path = join(args.output_dir, f"{base}_pred.png")
            Image.fromarray(pred_mask * 255).save(out_path)

    if dsc_list:
        mean_dsc = np.mean(dsc_list)
        std_dsc = np.std(dsc_list)
        hd_arr = np.asarray(hd95_list, dtype=float)
        hd_valid = hd_arr[~np.isnan(hd_arr)]
        if hd_valid.size > 0:
            mean_hd = float(np.mean(hd_valid))
            std_hd = float(np.std(hd_valid))
            hd_str = f"{mean_hd:.4f} ± {std_hd:.4f} (有效 {hd_valid.size}/{len(hd_arr)})"
        else:
            mean_hd, std_hd = float('nan'), float('nan')
            hd_str = "nan（无有效样本，请检查 pred/gt 是否全为空或 HD95 计算异常）"
        print(f"\n===== 评估结果 =====")
        print(f"样本数: {len(dsc_list)}")
        print(f"Mean DSC:  {mean_dsc:.4f} ± {std_dsc:.4f}")
        print(f"Mean HD95: {hd_str}")

        # 保存 CSV
        import csv
        csv_path = join(args.output_dir, "eval_results.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["File", "DSC", "HD95"])
            for name, dsc, hd in results:
                writer.writerow([name, f"{dsc:.6f}", f"{hd:.6f}"])
            writer.writerow([])
            writer.writerow(["Mean", f"{mean_dsc:.6f}", f"{mean_hd:.6f}"])
            writer.writerow(["Std", f"{std_dsc:.6f}", f"{std_hd:.6f}"])
        print(f"详细结果已保存至: {csv_path}")
    else:
        print("无有效评估结果。")


if __name__ == "__main__":
    main()
