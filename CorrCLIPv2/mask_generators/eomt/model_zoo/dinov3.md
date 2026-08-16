# EoMT Model Zoo - DINOv3

**Please note**: The reported FPS numbers for the DINOv3 models are not directly comparable to those of the [DINOv2](dinov2.md) models, due to differences in drivers, PyTorch version, and other software updates that were done in the meantime.

> FPS measured on NVIDIA H100 (torch.compile default / torch.compile with max-autotune).

## Panoptic Segmentation

### COCO

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">Input size</th>
<th valign="bottom">FPS</th>
<th valign="bottom">PQ</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: EoMT-S 640x640 -->
<tr><td align="left"><a href="../configs/dinov3/coco/panoptic/eomt_small_640_2x.yaml">EoMT-S</a><sup>2x</sup></td>
<td align="center">640×640</td>
<td align="center">246 / 570 </td>
<td align="center">47.2</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_panoptic_eomt_small_640_dinov3/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-B 640x640 -->
<tr><td align="left"><a href="../configs/dinov3/coco/panoptic/eomt_base_640_2x.yaml">EoMT-B</a><sup>2x</sup></td>
<td align="center">640×640</td>
<td align="center">240 / 347</td>
<td align="center">53.1</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_panoptic_eomt_base_640_dinov3/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-L 640x640 -->
<tr><td align="left"><a href="../configs/dinov3/coco/panoptic/eomt_large_640.yaml">EoMT-L</a></td>
<td align="center">640×640</td>
<td align="center">137 / 157</td>
<td align="center">56.8</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_panoptic_eomt_large_640_dinov3/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-L 1280x1280 -->
<tr><td align="left"><a href="../configs/dinov3/coco/panoptic/eomt_large_1280.yaml">EoMT-L</a></td>
<td align="center">1280×1280</td>
<td align="center">33 / 34</td>
<td align="center">58.9</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_panoptic_eomt_large_1280_dinov3/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
</tbody></table>  

*<sup><sup>2x</sup> Longer training schedule.</sup>*

## Semantic Segmentation

### ADE20K

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">Input size</th>
<th valign="bottom">FPS</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: EoMT-L 512x512 -->
<tr><td align="left"><a href="../configs/dinov3/ade20k/semantic/eomt_large_512.yaml">EoMT-L</a></td>
<td align="center">512×512</td>
<td align="center">137 / 200</td>
<td align="center">59.5</td>
<td align="center"><a href="https://huggingface.co/tue-mps/ade_semantic_eomt_large_512_dinov3/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
</tbody></table>

> FPS computed on random 512x512 inputs.

## Instance Segmentation

### COCO

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">Input size</th>
<th valign="bottom">FPS</th>
<th valign="bottom">mAP</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: EoMT-L 640x640 -->
<tr><td align="left"><a href="../configs/dinov3/coco/instance/eomt_large_640.yaml">EoMT-L</a></td>
<td align="center">640×640</td>
<td align="center">133 / 157</td>
<td align="center">45.9</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_instance_eomt_large_640_dinov3/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-L 1280x1280 -->
<tr><td align="left"><a href="../configs/dinov3/coco/instance/eomt_large_1280.yaml">EoMT-L</a></td>
<td align="center">1280×1280</td>
<td align="center">33 / 34</td>
<td align="center">49.9</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_instance_eomt_large_1280_dinov3/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
</tbody></table>

---

**Important:** The provided model weights are deltas with respect to DINOv3 weights. Users need to obtain access to the original DINOv3 weights first before using these models.
