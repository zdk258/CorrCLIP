<div align="center">

# CorrCLIP

**Training-free open-vocabulary semantic segmentation by reconstructing CLIP's patch correlations**

[![Paper](https://img.shields.io/badge/ArXiv-2411.10086-red?style=flat-square)](https://arxiv.org/abs/2411.10086)

</div>

This repository hosts two methods. Each subdirectory is self-contained — its own README, configs, dependencies and evaluation scripts — so pick one and work entirely inside it.

| Directory | Method | Status |
|---|---|---|
| [**CorrCLIPv1/**](CorrCLIPv1) | CorrCLIP | ICCV 2025 **Oral** |
| [**CorrCLIPv2/**](CorrCLIPv2) | CorrCLIPv2 | Journal extension, under review |

## CorrCLIP

CLIP's patch correlations are noisy: patches attend across class boundaries, which blurs the segmentation map. CorrCLIP reconstructs both the *scope* and the *value* of those correlations to suppress inter-class attention, strengthens the final patch features with two auxiliary branches, and updates the segmentation map with class-agnostic masks for spatial consistency. No training, no annotation.

→ [CorrCLIPv1/README.md](CorrCLIPv1/README.md) · [Colab demo](https://colab.research.google.com/github/zdk258/CorrCLIP/blob/master/CorrCLIPv1/corrclip_demo.ipynb)

## CorrCLIPv2

CorrCLIPv2 keeps that single-image pipeline and adds cross-image semantic support, retrieved from an offline memory bank built purely from unlabeled images — still no annotation, no caption, no training. Two components:

- **VISR** (VFM-Indexed Semantic Retrieval) builds the bank. A mask generator produces class-agnostic regions for each unlabeled image, and mask average pooling gives one embedding per region from both a vision foundation model and CorrCLIP. The VFM embeddings become the FAISS index keys, the CorrCLIP embeddings the semantic values. At inference each test region retrieves its Top-K neighbors and the returned values are fused with the image's own prediction — retrieval happens in VFM space, decoupled from text and vocabulary.
- **DMCE** (Decoupled Multi-CLIP Ensemble) stores region features from several CLIP variants under that one index, so auxiliary CLIP representations are *retrieved* rather than recomputed — ensemble gains at a few milliseconds of overhead.

Pre-generated masks and prebuilt memory banks are published so the results can be reproduced without building a bank: [dk258/CorrCLIPv2](https://huggingface.co/datasets/dk258/CorrCLIPv2).

→ [CorrCLIPv2/README.md](CorrCLIPv2/README.md)

## Citation

The journal extension is under review; please cite the conference paper for now:

```bibtex
@article{zhang2024corrclip,
  title={Corrclip: Reconstructing patch correlations in clip for open-vocabulary semantic segmentation},
  author={Zhang, Dengke and Liu, Fagui and Tang, Quan},
  journal={arXiv preprint arXiv:2411.10086},
  year={2024}
}
```
