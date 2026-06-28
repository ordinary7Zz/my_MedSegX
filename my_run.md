# Ubuntu 上的完整运行流程

## 1. 准备模型权重目录

```bash
mkdir -p playground/SAM
mkdir -p playground/MedSegX
```

把下面两个文件放到对应目录：

- SAM 基础权重：`playground/SAM/sam_vit_b_01ec64.pth`
- MedSegX 预训练权重：`playground/MedSegX/medsegx_vit_b.pth`

> **checkpoint 与 model_weight 的区别**：SAM 权重是基础视觉模型的 backbone，MedSegX 权重是在此基础上针对医学分割任务预训练过的完整模型参数。两者缺一不可。

---

## 2. 相关脚本功能说明

| 脚本 | 功能 |
|------|------|
| `scripts/convert_image_to_npy.py` | 把原始图片和 mask 批量转换成 `.npy`，自动生成 `npy_imgs/` 和 `npy_gts/` 目录结构 |
| `infer_simple.py` | **简单推理**：输入图像目录 + 权重路径，输出所有图像的掩码（PNG） |
| `infer_external_fullimg.py` | **完整推理 + 评估**：输入 `.npy` 目录，推理并计算 DSC/HD 指标 |
| `evaluate_external.py` | 批量外部评估，遍历多个数据集并汇总结果 |
| `finetune.py` | 在预训练权重基础上，对特定数据集微调 |
| `scripts/finetune-site.sh` | 跨中心微调（cross_site）的快捷脚本 |
| `scripts/finetune-task.sh` | 跨任务微调（cross_task）的快捷脚本 |
| `scripts/thyroid-realworld-infer.sh` | 批量运行甲状腺 RealWorld 外部评估，并计算 95% 置信区间 |

---

## 3. 数据准备：PNG/JPG → NPY

如果原始数据是图片文件，先用转换脚本批量生成 MedSegX 需要的目录结构，并统一 resize 到 `224 224`。

### 路径模板

```
playground/{DatasetName}/eval/RealWorld/cross_site/{TaskName}/{DatasetName}/{子目录}/
```

### 示例：甲状腺腺体分割（`US_GlndThyroid`）

```bash
# 推理数据
python scripts/convert_image_to_npy.py \
    --images /mnt/wangbd8/workspace/DataSets/ThyroidAgent/TGVideo_PNG/train/image \
    --masks /mnt/wangbd8/workspace/DataSets/ThyroidAgent/TGVideo_PNG/train/mask \
    --output_root ./playground/TG_Video/eval/RealWorld/cross_site/US_GlndThyroid/TG_Video/finetune \
    --size 224 224
```

### 示例：甲状腺结节分割（`US_ThyroidNodule`）

```bash
# 推理数据
python scripts/convert_image_to_npy.py \
    --images /path/to/nodule/images \
    --masks /path/to/nodule/masks \
    --output_root ./playground/NoduleData/eval/RealWorld/cross_site/US_ThyroidNodule/NoduleData/inference \
    --size 224 224
```

转换完成后目录结构：

```text
inference/
├── npy_imgs/
└── npy_gts/
```

---

## 4. 训练（微调）

### 4.1 准备微调数据目录

微调需要的是 `finetune` 子目录（不是 `inference`），结构和推理一致：

#### 甲状腺腺体（`US_GlndThyroid`）

```
playground/GlndData/eval/RealWorld/cross_site/US_GlndThyroid/GlndData/
└── finetune/
    ├── npy_imgs/    ← 训练图像 .npy
    └── npy_gts/     ← 训练标注 .npy
```

#### 甲状腺结节（`US_ThyroidNodule`）

```
playground/NoduleData/eval/RealWorld/cross_site/US_ThyroidNodule/NoduleData/
└── finetune/
    ├── npy_imgs/    ← 训练图像 .npy
    └── npy_gts/     ← 训练标注 .npy
```

### 4.2 运行微调

```bash
# 甲状腺腺体微调
python finetune.py \
    --checkpoint ./playground/SAM \
    --model_type vit_b \
    --data_path ./playground/GlndData/eval/RealWorld \
    --shift_type cross_site \
    --resume ./playground/MedSegX/medsegx_vit_b.pth \
    --num_epochs 30 \
    --batch_size 64 \
    --lr 5e-5 \
    --validation val \
    --use_amp

# 甲状腺结节微调
python finetune.py \
    --checkpoint ./playground/SAM \
    --model_type vit_b \
    --data_path ./playground/NoduleData/eval/RealWorld \
    --shift_type cross_site \
    --resume ./playground/MedSegX/medsegx_vit_b.pth \
    --num_epochs 30 \
    --batch_size 64 \
    --lr 5e-5 \
    --validation val \
    --use_amp
```

| 参数 | 说明 |
|------|------|
| `--checkpoint` | SAM backbone 权重所在目录 |
| `--model_type` | SAM 模型规模：`vit_b` / `vit_l` / `vit_h` |
| `--data_path` | 数据根目录，代码会自动遍历其下所有 task 和 dataset |
| `--shift_type` | `cross_site`（跨中心）或 `cross_task`（跨任务） |
| `--resume` | 预训练的 MedSegX 权重路径，作为微调起点 |
| `--num_epochs` | 训练轮数，默认 30 |
| `--batch_size` | 批次大小，默认 64 |
| `--lr` | 学习率，微调推荐 `5e-5` |
| `--validation` | `val`（从微调数据切 80/20）或 `test`（用 inference 目录） |

**关于 task_name**：根据 `data/datainfo.py` 中的定义：
- 甲状腺腺体分割：`US_GlndThyroid`
- 甲状腺结节分割：`US_ThyroidNodule`

### 4.3 权重保存位置

```
playground/MedSegX/finetune/cross_site/{task}/{dataset}/
├── model_best.pth      ← 最佳 DSC 权重
├── train_loss.png      ← 训练 loss 曲线
├── val_dsc.png         ← 验证 DSC 曲线
├── lr.png              ← 学习率曲线
└── result.log          ← 每轮详细日志
```

---

## 5. 推理

### 5.1 简单推理（输入图片目录，输出掩码 PNG）

```bash
# 甲状腺腺体推理
python infer_simple.py \
    --input_dir /path/to/glnd/test_images \
    --output_dir /path/to/output/glnd_masks \
    --task_name US_GlndThyroid \
    --checkpoint ./playground/SAM \
    --model_weight ./playground/MedSegX/finetune/cross_site/US_GlndThyroid/GlndData/model_best.pth \
    --device cuda:0

# 甲状腺结节推理
python infer_simple.py \
    --input_dir /path/to/nodule/test_images \
    --output_dir /path/to/output/nodule_masks \
    --task_name US_ThyroidNodule \
    --checkpoint ./playground/SAM \
    --model_weight ./playground/MedSegX/finetune/cross_site/US_ThyroidNodule/NoduleData/model_best.pth \
    --device cuda:0
```

支持 PNG/JPG/BMP/TIFF 等常见格式，输出为二值 PNG 掩码。

### 5.2 完整评估推理（需要 .npy + ground truth）

```bash
bash scripts/thyroid-realworld-infer.sh
```

---

## 6. 结果输出目录

```bash
ls playground/MedSegX/external
```

结果文件示例：
- `playground/MedSegX/external/{Dataset}-RealWorld-site.md`
- `playground/MedSegX/external/{Dataset}-RealWorld-site.csv`
- `playground/MedSegX/external/thyroid_ci_summary.csv`
