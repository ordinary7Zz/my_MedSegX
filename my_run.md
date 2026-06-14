# Ubuntu 上的完整运行流程

## 1. 进入项目目录

```bash
cd /mnt/wangbd8/workspace/ThyroidAgent/MedSegX-code
```

## 2. 创建并激活虚拟环境（conda）

```bash
conda create -n medsegx python=3.10 -y
conda activate medsegx
```

## 3. 安装 PyTorch 与 CUDA 依赖

```bash
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

## 4. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

## 5. 准备模型权重目录

```bash
mkdir -p playground/SAM
mkdir -p playground/MedSegX
```

## 6. 下载并放置权重文件

把下面两个文件放到对应目录：

- SAM 基础权重：`playground/SAM/sam_vit_b_01ec64.pth`
- MedSegX 权重：`playground/MedSegX/medsegx_vit_b.pth`

## 7. 相关脚本功能说明

- `scripts/convert_image_to_npy.py`：把原始图片和 mask 批量转换成 `.npy`，并自动生成 MedSegX 需要的 `npy_imgs/` 和 `npy_gts/` 目录结构。它支持单文件转换，也支持 `--images` + `--masks` 的配对批量转换。
- `scripts/thyroid-realworld-infer.sh`：批量运行甲状腺 RealWorld 外部评估，依次检查各数据集的 `inference/npy_imgs/` 和 `inference/npy_gts/`，再调用 `evaluate_external.py` 做推理并汇总结果与 95% 置信区间。

## 8. 如果原始数据还是 JPG / PNG，先转换成 NPY

如果你现在手里的原始数据是图片文件和 mask 文件，可以先用仓库里的转换脚本批量生成 MedSegX 需要的目录结构：

```bash
python scripts/convert_image_to_npy.py \
    --images /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/finall_data/image \
    --masks /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/finall_data/mask \
    --output_root ./playground/FinalData/eval/RealWorld/cross_site/US_ThyroidNodule/FinalData/inference

python scripts/convert_image_to_npy.py \
    --images /mnt/wangbd8/workspace/DataSets/ThyroidAgent/augtrain_PNG/image \
    --masks /mnt/wangbd8/workspace/DataSets/ThyroidAgent/augtrain_PNG/mask \
    --output_root ./playground/Augtrain/eval/RealWorld/cross_site/US_ThyroidNodule/Augtrain/inference
```

转换完成后会得到：

```text
/path/to/inference/
├── npy_imgs/
└── npy_gts/
```

## 8. 确认 5 个数据集目录存在

```bash
ls /mnt/wangbd8/workspace/ThyroidAgent/MedSegX-code/playground/DDTI/eval/RealWorld/cross_site/US_ThyroidNodule/DDTI/inference
ls /mnt/wangbd8/workspace/ThyroidAgent/MedSegX-code/playground/PKTN/eval/RealWorld/cross_site/US_ThyroidNodule/PKTN/inference
ls /mnt/wangbd8/workspace/ThyroidAgent/MedSegX-code/playground/ThyroidXL/eval/RealWorld/cross_site/US_ThyroidNodule/ThyroidXL/inference
ls /mnt/wangbd8/workspace/ThyroidAgent/MedSegX-code/playground/TN3K/eval/RealWorld/cross_site/US_ThyroidNodule/TN3K/inference
ls /mnt/wangbd8/workspace/ThyroidAgent/MedSegX-code/playground/TN5K/eval/RealWorld/cross_site/US_ThyroidNodule/TN5K/inference
```

每个 `inference` 目录下应包含：

- `npy_imgs/`
- `npy_gts/`

## 9. 给推理脚本执行权限

```bash
chmod +x scripts/thyroid-realworld-infer.sh
```

## 9. 运行推理并自动计算 DSC 和 HD95 的 95% 置信区间

```bash
bash scripts/thyroid-realworld-infer.sh
```

## 10. 单卡 0 号 GPU 运行

```bash
DEVICE_IDS="0" bash scripts/thyroid-realworld-infer.sh
```

## 11. 多卡运行示例

```bash
DEVICE_IDS="0 1" bash scripts/thyroid-realworld-infer.sh
```

## 12. 自定义 batch size 示例

```bash
DEVICE_IDS="0" BATCH_SIZE="16" bash scripts/thyroid-realworld-infer.sh
```

## 13. 结果输出目录

```bash
ls playground/MedSegX/external
```

结果文件示例：

- `playground/MedSegX/external/DDTI-RealWorld-site.md`
- `playground/MedSegX/external/DDTI-RealWorld-site.csv`
- `playground/MedSegX/external/PKTN-RealWorld-site.md`
- `playground/MedSegX/external/PKTN-RealWorld-site.csv`
- `playground/MedSegX/external/ThyroidXL-RealWorld-site.md`
- `playground/MedSegX/external/ThyroidXL-RealWorld-site.csv`
- `playground/MedSegX/external/TN3K-RealWorld-site.md`
- `playground/MedSegX/external/TN3K-RealWorld-site.csv`
- `playground/MedSegX/external/TN5K-RealWorld-site.md`
- `playground/MedSegX/external/TN5K-RealWorld-site.csv`
- `playground/MedSegX/external/thyroid_ci_summary.csv`
