<div align="center">

# CorrCLIP

**Training-free open-vocabulary semantic segmentation by reconstructing CLIP's patch correlations**

<p>
  <a href="https://arxiv.org/abs/2411.10086"><img src="https://img.shields.io/badge/CorrCLIP-ICCV.2025-007EC6?style=flat&amp;labelColor=555555" alt="CorrCLIP | ICCV.2025" height="20"></a>
  &nbsp;&nbsp;
  <a href="https://link.springer.com/article/10.1007/s11263-026-03027-3"><img src="https://img.shields.io/badge/CorrCLIPv2-IJCV.2026-800080?style=flat&amp;labelColor=555555" alt="CorrCLIPv2 | IJCV.2026" height="20"></a>
  &nbsp;&nbsp;
  <a href="https://arxiv.org/abs/2610.24757"><img src="https://img.shields.io/badge/CorrCLIPv2-arXiv.2610-BB4444?style=flat&amp;labelColor=555555" alt="CorrCLIPv2 | arXiv.2610" height="20"></a>
</p>

</div>

This repository hosts two methods. Each subdirectory is self-contained — its own README, configs, dependencies and evaluation scripts — so pick one and work entirely inside it.

| Method | Publication | Code & Docs |
| :--- | :---: | :---: |
| **CorrCLIP** | ICCV 2025 (Oral) | [CorrCLIPv1](CorrCLIPv1/README.md) |
| **CorrCLIPv2** | IJCV 2026 | [CorrCLIPv2](CorrCLIPv2/README.md) |

## CorrCLIP

CorrCLIP enables training-free open-vocabulary semantic segmentation by reconstructing the scope and value of CLIP's patch correlations to suppress interactions between different classes. It uses SAM masks to constrain patch interactions and self-supervised DINO features to compute more semantically coherent attention scores. Spatial and semantic refinement branches enrich patch features, while voting within each mask improves the spatial consistency of the final segmentation.

## CorrCLIPv2

CorrCLIPv2 extends CorrCLIP with cross-image object retrieval from unlabeled images to reduce contextual bias while remaining training-free. VFM-Indexed Semantic Retrieval (VISR) uses vision foundation model features to retrieve semantically similar objects from an offline memory bank and aggregates their CorrCLIP features to strengthen predictions across diverse contexts. Decoupled Multi-CLIP Ensemble (DMCE) combines complementary representations from multiple CLIP models through the shared retrieval index, moving auxiliary feature encoding offline to improve robustness with minimal inference overhead.

The arXiv version corresponds to the initial submission, while the journal version includes additional ablation studies and re-evaluates other methods using the same models as CorrCLIPv2 for a consistent comparison.

## Citation

If you use CorrCLIP or CorrCLIPv2 in your research, please cite:

```bibtex
@InProceedings{Zhang_2025_ICCV,
    author    = {Zhang, Dengke and Liu, Fagui and Tang, Quan},
    title     = {CorrCLIP: Reconstructing Patch Correlations in CLIP for Open-Vocabulary Semantic Segmentation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {24677-24687}
}

@article{zhang2026corrclipv2,
  title={CorrCLIPv2: Harnessing Unlabeled Images for Open-Vocabulary Semantic Segmentation},
  author={Zhang, Dengke and Liu, Fagui and Tang, Quan and Chen, CL Philip},
  journal={International Journal of Computer Vision},
  volume={134},
  number={11},
  pages={478},
  year={2026},
  publisher={Springer}
}
```
