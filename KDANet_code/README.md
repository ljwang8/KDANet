# KDANet

PyTorch implementation of KDANet for lightweight mining-area remote sensing scene classification.

This repository provides the core implementation of the method described in:

**A Lightweight Mining-Area Remote Sensing Scene Classification Framework via Knowledge Distillation and Channel-Aware Non-Local Attention**

The code focuses on the proposed method only. It includes the ShuffleNetV2 student network, the channel-aware non-local attention module (CANLAM), and the multi-perspective distillation objective composed of contrastive, relational, feature, and classification losses.

## Project Structure

```text
KDANet/
├── configs/
│   └── kdanet.yaml
├── kdanet/
│   ├── data.py
│   ├── losses/
│   ├── models/
│   └── utils/
├── tools/
│   └── check_dataset.py
├── train.py
├── test.py
└── requirements.txt
```

## Dataset Layout

The default data loader uses the standard `ImageFolder` directory format:

```text
CUG_MA/
├── train/
│   ├── Mining Area/
│   ├── Coal Yard/
│   └── ...
├── val/
│   ├── Mining Area/
│   ├── Coal Yard/
│   └── ...
└── test/
    ├── Mining Area/
    ├── Coal Yard/
    └── ...
```

The manuscript uses nine scene categories:

```text
Mining Area
Coal Yard
Hillock
Mining Architecture
Gangue Dump
Tailing Pond
Refuse Dump
Mineral Processing Area
Transfer Site
```

## Teacher Model

KDANet is trained in a teacher-student manner. The paper uses HTransNet as the teacher model. Since the complete HTransNet architecture is defined in the previous HTransNet work, this repository exposes a teacher adapter rather than hard-coding an incomplete teacher implementation.

For distillation training, the teacher model should return either:

```python
{"logits": logits, "features": feature_map}
```

or:

```python
(logits, feature_map)
```

where `logits` has shape `[B, num_classes]`, and `feature_map` has shape `[B, C, H, W]`.

## Training

Edit `configs/kdanet.yaml`, then run:

```bash
python train.py --config configs/kdanet.yaml --teacher-checkpoint path/to/htransnet_teacher.pt
```

If the teacher architecture is available in another Python file, provide a factory function:

```bash
python train.py \
  --config configs/kdanet.yaml \
  --teacher-factory tools.teacher_adapter_example:build_teacher \
  --teacher-checkpoint path/to/htransnet_teacher.pt
```

The command above is only an example. Replace `tools.teacher_adapter_example:build_teacher` with the actual HTransNet constructor in your project.

For quick verification without a teacher model, run:

```bash
python train.py --config configs/kdanet.yaml
```

In this mode, only the classification loss is used.

Important default settings follow the manuscript:

- epochs: `200`
- batch size: `16`
- optimizer: SGD
- learning rate: `0.001`
- momentum: `0.9`
- weight decay: `0.0001`
- loss weights: contrastive `0.10`, relational `0.10`, feature `0.10`, classification `1.0`

## Evaluation

```bash
python test.py --config configs/kdanet.yaml --checkpoint runs/kdanet/best.pth
```

The evaluation script reports OA, AA, Kappa, and per-class precision, recall, and F1-score.

## Notes

- Large datasets and checkpoints are not included in this repository.
- The default implementation is intended for research and reproducibility.
- The teacher checkpoint must be supplied separately for full distillation training.
