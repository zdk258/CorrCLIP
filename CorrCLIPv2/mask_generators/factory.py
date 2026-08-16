"""Instance mask generator factory: uniform wrappers for entityseg / eomt / sam / sam2.

Memory-bank building (memory_bank_builder/generate_mask.py) and deployment inference
(the mask_generator of corrclip_segmentor.py) share this code; the two sides use
different default hyper-parameters (build-time vs deployment), passed explicitly by
each caller.

Uniform interface:
    gen = build_mask_generator(name, device, **kwargs)
    instance_mask = gen.generate(img_path)   # [H, W] int tensor; 0 = unsegmented, >0 = instance id
"""
import contextlib

import numpy as np
import torch
from PIL import Image

MASK_GENERATOR_CHOICES = ('entityseg', 'eomt', 'sam', 'sam2')

_EXIF_ORIENT = 274  # exif 'Orientation' tag


def _apply_exif_orientation(image):
    """Byte-for-byte port of detectron2.data.detection_utils._apply_exif_orientation,
    so masks align with the CLIP/VFM feature steps (which read images through
    detectron2). Reimplemented locally to avoid a hard detectron2 dependency."""
    if not hasattr(image, "getexif"):
        return image
    try:
        exif = image.getexif()
    except Exception:
        exif = None
    if exif is None:
        return image
    orientation = exif.get(_EXIF_ORIENT)
    method = {
        2: Image.FLIP_LEFT_RIGHT,
        3: Image.ROTATE_180,
        4: Image.FLIP_TOP_BOTTOM,
        5: Image.TRANSPOSE,
        6: Image.ROTATE_270,
        7: Image.TRANSVERSE,
        8: Image.ROTATE_90,
    }.get(orientation)
    if method is not None:
        return image.transpose(method)
    return image


def _autocast(dtype):
    return torch.autocast('cuda', dtype=dtype) if dtype is not None else contextlib.nullcontext()


def _load_pil(img_path, apply_exif):
    pil_img = Image.open(img_path).convert('RGB')
    if apply_exif:
        pil_img = _apply_exif_orientation(pil_img)
    return pil_img


def _anns_to_instance_mask(anns, hw, device):
    """SAM/SAM2 anns -> instance-id map. Painted in ascending-area order, so larger
    instances painted later overwrite smaller ones."""
    instance_mask = np.zeros(hw, dtype=int)
    if len(anns) != 0:
        for instance_id, ann in enumerate(sorted(anns, key=lambda x: x['area']), start=1):
            instance_mask[ann['segmentation']] = instance_id
    return torch.from_numpy(instance_mask).to(device)


class EntitySegGenerator:
    """CropFormer (EntitySegV2, hornet-3x). Reads images via detectron2's read_image,
    which applies EXIF correction."""

    def __init__(self, device, autocast_dtype=torch.float16, conf_thresh=0.5,
                 cfg_file='mask2former_hornet_3x.yaml',
                 ckpt_path='Mask2Former_hornet_3x_576d0b.pth'):
        from CropFormer.demo_mask2former.demo import get_entityseg
        self.device = device
        self.autocast_dtype = autocast_dtype
        self.conf_thresh = conf_thresh
        self.predictor = get_entityseg(cfg_file=cfg_file, ckpt_path=ckpt_path)

    def generate(self, img_path):
        from detectron2.data.detection_utils import read_image
        img = read_image(img_path, format='BGR')
        with torch.inference_mode(), _autocast(self.autocast_dtype):
            predictions = self.predictor(img)

        pred_masks = predictions['instances'].pred_masks
        pred_scores = predictions['instances'].scores
        selected = pred_scores >= self.conf_thresh
        selected_scores = pred_scores[selected]
        selected_masks = pred_masks[selected]
        _, m_H, m_W = selected_masks.shape
        instance_mask = torch.zeros((m_H, m_W), dtype=torch.int, device=self.device)

        # Preserved from the original implementation: stored id = original index + 1 (not rank);
        # visiting in ascending-score order lets high-score masks overwrite low-score ones
        selected_scores, ranks = torch.sort(selected_scores)
        ranks = ranks + 1
        for index in ranks:
            instance_mask[(selected_masks[index - 1] == 1)] = int(index)
        return instance_mask


class EoMTGenerator:
    """EoMT (COCO-panoptic pretrained; weights auto-downloaded from Hugging Face and cached)."""

    def __init__(self, device, autocast_dtype=torch.float16, apply_exif=False,
                 cfg_file='coco_panoptic_eomt_large_640', use_compile=False):
        from eomt.infer import get_eomt
        self.device = device
        self.autocast_dtype = autocast_dtype or torch.float16  # EoMT forward always runs under autocast
        self.apply_exif = apply_exif
        self.model = get_eomt(cfg_file=cfg_file, use_compile=use_compile).to(device)

    def generate(self, img_path):
        import torch.nn.functional as F
        from torchvision import tv_tensors
        img = tv_tensors.Image(_load_pil(img_path, self.apply_exif))
        with torch.inference_mode(), torch.autocast('cuda', dtype=self.autocast_dtype):
            imgs = [img.to(self.device)]
            img_sizes = [im.shape[-2:] for im in imgs]

            transformed_imgs = self.model.resize_and_pad_imgs_instance_panoptic(imgs)
            mask_logits_per_layer, class_logits_per_layer = self.model(transformed_imgs)
            mask_logits = F.interpolate(mask_logits_per_layer[-1], self.model.img_size, mode='bilinear')
            mask_logits = self.model.revert_resize_and_pad_logits_instance_panoptic(mask_logits, img_sizes)

            preds = self.model.to_per_pixel_preds_panoptic(
                mask_logits,
                class_logits_per_layer[-1],
                self.model.stuff_classes,
                self.model.mask_thresh,
                self.model.overlap_thresh,
            )[0]
        return preds[..., 1]


class SAMGenerator:
    """SAM (segment-anything). Default ViT-H weights at data/sam_vit_h_4b8939.pth."""

    def __init__(self, device, autocast_dtype=None, apply_exif=False,
                 model_type='vit_h', checkpoint='data/sam_vit_h_4b8939.pth',
                 pred_iou_thresh=0.7, stability_score_thresh=0.7):
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        self.device = device
        self.autocast_dtype = autocast_dtype
        self.apply_exif = apply_exif
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=device)
        self.generator = SamAutomaticMaskGenerator(
            sam,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
        )

    def generate(self, img_path):
        img = np.array(_load_pil(img_path, self.apply_exif))
        with torch.inference_mode(), _autocast(self.autocast_dtype):
            anns = self.generator.generate(img)
        return _anns_to_instance_mask(anns, img.shape[:2], self.device)


class SAM2Generator:
    """SAM2 (hiera-large). Default weights at data/sam2_hiera_large.pt."""

    def __init__(self, device, autocast_dtype=None, apply_exif=False,
                 cfg_file='sam2_hiera_l.yaml', checkpoint='data/sam2_hiera_large.pt',
                 model_dtype=None, points_per_side=None,
                 pred_iou_thresh=0.7, stability_score_thresh=0.7, multimask_output=True):
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        self.device = device
        self.autocast_dtype = autocast_dtype
        self.apply_exif = apply_exif
        model = build_sam2(cfg_file, checkpoint, device=device, apply_postprocessing=False)
        if model_dtype is not None:
            model = model.to(model_dtype)
        kwargs = dict(
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            multimask_output=multimask_output,
        )
        if points_per_side is not None:
            kwargs['points_per_side'] = points_per_side
        self.generator = SAM2AutomaticMaskGenerator(model=model, **kwargs)

    def generate(self, img_path):
        img = np.array(_load_pil(img_path, self.apply_exif))
        with torch.inference_mode(), _autocast(self.autocast_dtype):
            anns = self.generator.generate(img)
        return _anns_to_instance_mask(anns, img.shape[:2], self.device)


_GENERATORS = {
    'entityseg': EntitySegGenerator,
    'eomt': EoMTGenerator,
    'sam': SAMGenerator,
    'sam2': SAM2Generator,
}


def build_mask_generator(name, device, **kwargs):
    if name not in _GENERATORS:
        raise ValueError(f"Unsupported mask generator: {name} (choices: {', '.join(MASK_GENERATOR_CHOICES)})")
    return _GENERATORS[name](device, **kwargs)
