# -*- coding: utf-8 -*-
"""
MedSegX external inference with full-image box prompts.
Ground-truth masks are only used for evaluation.
"""

import argparse
import os
join = os.path.join

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
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
from utils.metric import SegmentMetrics


parser = argparse.ArgumentParser("MedSegX external full-image-box evaluation", add_help=False)
# model
parser.add_argument("--checkpoint", type=str, default="./playground/SAM",
                    help="path to SAM checkpoint folder")
parser.add_argument("--model_type", type=str, default="vit_b",
                    help="SAM model scale (e.g vit_b, vit_l, vit_h)")
parser.add_argument("--model_weight", type=str, default="./playground/MedSegX/medsegx_vit_b.pth",
                    help="path to MedSegX model weight")
parser.add_argument("--method", type=str, default="medsegx")
parser.add_argument("--bottleneck_dim", type=int, default=16)
parser.add_argument("--embedding_dim", type=int, default=16)
parser.add_argument("--expert_num", type=int, default=4)
# data
parser.add_argument("--input_dir", type=str, required=True,
                    help="path to the inference folder containing npy_imgs and npy_gts")
parser.add_argument("--task_name", type=str, required=True,
                    help="task name in modal_organ format, e.g. US_ThyroidNodule")
parser.add_argument("--output_dir", type=str, required=True,
                    help="directory to save predictions and evaluation")
parser.add_argument("--metric", type=str, default=["dsc", "hd"], nargs='+',
                    help="evaluation metrics (e.g dsc, hd)")
# infer
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=8)


class TaskImageEvalDataset(Dataset):
    def __init__(self, input_dir, task_name):
        self.input_dir = input_dir
        self.task_name = task_name
        self.img_dir = join(input_dir, "npy_imgs")
        self.gt_dir = join(input_dir, "npy_gts")
        files = sorted(os.listdir(self.img_dir))
        self.file_names = [join(self.img_dir, f) for f in files if f.endswith(".npy")]

        modal = task_name.split('_')[0]
        organ = ('').join(task_name.split('_')[1:]).rstrip('0123456789')

        self.modal = modal_map[modal_dict[modal]]
        organ_level_1 = next(organ_level_1_map[k] for k, v in organ_level_1_dict.items() if organ in v)
        organ_level_2 = next(organ_level_2_map[k] for k, v in organ_level_2_dict.items() if organ in v)
        organ_level_3 = next(organ_level_3_map[k] for k, v in organ_level_3_dict.items() if organ in v)
        organ_level_4 = task_idx[organ]
        self.organ = (organ_level_1, organ_level_2, organ_level_3, organ_level_4)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_name = self.file_names[index]
        gt_name = join(self.gt_dir, os.path.basename(file_name))

        img = np.load(file_name).transpose(2, 0, 1)
        gt = np.load(gt_name)
        _, h, w = img.shape
        box = np.array([0, 0, w, h], dtype=np.float32)

        data = {
            "img": torch.tensor(img).float(),
            "box": torch.tensor(box).float(),
            "modal": self.modal,
            "organ": self.organ,
            "name": file_name,
        }
        return data, torch.tensor(gt[None, :, :]).long()


def forward_with_iou(model, data):
    img, box = data["img"], data["box"]

    if len(box.shape) == 2:
        box = box[:, None, :]

    sparse_embeddings, dense_embeddings = model.sam.prompt_encoder(
        points=None,
        boxes=box,
        masks=None,
    )

    input_image = model.sam.preprocess(img)
    if isinstance(model, MedSegX):
        modal, organ = data["modal"], data["organ"]
        batch_size = img.shape[0]

        modal_index = model.sam.image_encoder.modal_index[modal]
        modal_embed = model.sam.image_encoder.modal_embed(modal_index)

        organ_1, organ_2, organ_3, organ_4 = organ
        organ_index_0 = torch.zeros(batch_size, dtype=torch.long, device=img.device)
        organ_embed_0 = model.sam.image_encoder.organ_embed[0](organ_index_0)
        organ_index_1 = model.sam.image_encoder.organ_index_1[organ_1]
        organ_embed_1 = model.sam.image_encoder.organ_embed[1](organ_index_1)
        organ_index_2 = model.sam.image_encoder.organ_index_2[organ_2]
        organ_embed_2 = model.sam.image_encoder.organ_embed[2](organ_index_2)
        organ_index_3 = model.sam.image_encoder.organ_index_3[organ_3]
        organ_embed_3 = model.sam.image_encoder.organ_embed[3](organ_index_3)
        organ_embed_4 = model.sam.image_encoder.organ_embed[4](organ_4)
        organ_embed = (organ_embed_0, organ_embed_1, organ_embed_2, organ_embed_3, organ_embed_4)

        image_embedding, _ = model.sam.image_encoder(input_image, modal_embed, organ_embed)
    elif isinstance(model, MedSAM):
        image_embedding = model.sam.image_encoder(input_image)
    else:
        raise NotImplementedError(f"Unsupported model type: {type(model).__name__}")

    mask_predictions, iou_predictions = model.sam.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=True,
    )
    return mask_predictions, iou_predictions


def run_evaluation(model, metric, dataloader, img_size, img_transform, box_transform, output_dir, args):
    model.eval()
    device = torch.device(args.device)
    pred_dir = join(output_dir, "npy_preds")
    os.makedirs(pred_dir, exist_ok=True)

    result_total = {m: [] for m in args.metric}
    rows = []

    pbar = tqdm(dataloader)
    pbar.set_description("Evaluating")
    with torch.no_grad():
        for data, label in pbar:
            if data["img"].shape[-1] != img_size:
                data["box"] = box_transform.apply_boxes_torch(
                    data["box"].reshape(-1, 2, 2),
                    data["img"].shape[-2:],
                ).reshape(-1, 4)
                data["img"] = img_transform(data["img"])

            data["img"] = data["img"].to(device, non_blocking=True)
            data["box"] = data["box"].to(device, non_blocking=True)
            data["modal"] = data["modal"].to(device, non_blocking=True)
            data["organ"] = tuple(v.to(device, non_blocking=True) for v in data["organ"])
            label = label.to(device, non_blocking=True, dtype=torch.bool)

            mask_pred, iou_pred = forward_with_iou(model, data)
            if mask_pred.shape[-2:] != label.shape[-2:]:
                mask_pred = F.interpolate(mask_pred, size=label.shape[-2:], mode="bilinear", antialias=True)

            mask_prob = torch.sigmoid(mask_pred)
            best_idx = iou_pred.argmax(dim=1)
            batch_index = torch.arange(mask_prob.shape[0], device=device)
            chosen_prob = mask_prob[batch_index, best_idx]
            chosen_mask = (chosen_prob > 0.5).bool().unsqueeze(1)
            best_score = iou_pred[batch_index, best_idx].cpu().numpy()
            best_idx_np = best_idx.cpu().numpy()

            metric_batch = metric(chosen_mask, label)
            metric_dict = {}
            for m in args.metric:
                result = metric_batch[m]
                result_total[m].append(result)
                metric_dict[m] = result.mean().item()

            pbar.set_postfix(metric_dict)

            chosen_mask_np = chosen_mask.squeeze(1).to(torch.uint8).cpu().numpy()
            for idx, name in enumerate(data["name"]):
                base_name = os.path.basename(name)
                out_path = join(pred_dir, base_name)
                np.save(out_path, chosen_mask_np[idx])
                row = {
                    "File": name,
                    "PredMask": out_path,
                    "SelectedMaskIndex": int(best_idx_np[idx]),
                    "PredIoU": float(best_score[idx]),
                }
                for m in args.metric:
                    row[m.upper()] = metric_batch[m][idx].item()
                rows.append(row)

    summary = {m.upper(): torch.cat(v).mean().item() for m, v in result_total.items()}
    summary_df = pd.DataFrame([{"N_CASE": len(rows), **summary}])
    summary_df.to_csv(join(output_dir, "summary.csv"), index=False)

    with open(join(output_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# external full-image-box evaluation\n\n")
        metrics_text = ", ".join([f"{k} ({v:.4f})" for k, v in summary.items()])
        f.write(f"- Mean: {metrics_text}\n")
        f.write(f"- Cases: {len(rows)}\n")

    pd.DataFrame(rows).to_csv(join(output_dir, "predictions.csv"), index=False)


def main(args):
    device = torch.device(args.device)
    checkpoint = join(args.checkpoint, sam_model_checkpoint[args.model_type])
    sam_model = sam_model_registry[args.model_type](image_size=256, keep_resolution=True, checkpoint=checkpoint)

    if args.method == "medsam":
        model = MedSAM(sam_model).to(device)
    elif args.method == "medsegx":
        model = MedSegX(sam_model, args.bottleneck_dim, args.embedding_dim, args.expert_num).to(device)
    else:
        raise NotImplementedError(f"Method {args.method} not implemented!")

    seg_metric = SegmentMetrics(args.metric).to(device)

    if os.path.isfile(args.model_weight):
        print(f"load model from {args.model_weight}")
        checkpoint = torch.load(args.model_weight, map_location=device)
        model.load_parameters(checkpoint["model"])
    else:
        raise FileNotFoundError(f"model weight {args.model_weight} not found!")

    dataset = TaskImageEvalDataset(args.input_dir, args.task_name)
    if len(dataset) == 0:
        raise RuntimeError(f"No .npy files found in {join(args.input_dir, 'npy_imgs')}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    img_size = model.sam.image_encoder.img_size
    img_transform = Resize((img_size, img_size), antialias=True)
    box_transform = ResizeLongestSide(img_size)

    run_evaluation(model, seg_metric, dataloader, img_size, img_transform, box_transform, args.output_dir, args)
    print(f"save predictions and evaluation to {args.output_dir}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
