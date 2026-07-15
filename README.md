# KDANet

Source code for the paper:

**A Lightweight Mining-Area Remote Sensing Scene Classification Framework via Knowledge Distillation and Channel-Aware Non-Local Attention**

KDANet is a lightweight mining-area remote sensing scene classification framework based on knowledge distillation and channel-aware non-local attention. The method uses HTransNet as the teacher network and ShuffleNetV2 as the student network, and improves the student model through feature-level, relational, contrastive, and classification losses.

## Main Components

- ShuffleNetV2 student network
- Channel-Aware Non-Local Attention Module (CANLAM)
- Feature distillation loss
- Relational distillation loss
- Contrastive distillation loss
- Classification loss

## Dataset

The experiments are conducted on the CUG_MA mining-area remote sensing scene classification dataset, which contains nine scene categories:

- Mining Area
- Coal Yard
- Hillock
- Mining Architecture
- Gangue Dump
- Tailing Pond
- Refuse Dump
- Mineral Processing Area
- Transfer Site

Dataset link: https://doi.org/10.5281/zenodo.15172547

## Environment

```bash
pip install -r requirements.txt
