import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
sys.path.append("..")

import numpy as np
from PIL import Image
from pathlib import Path

from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.registry import MODELS
from mmengine.structures import PixelData

from torchvision import transforms, tv_tensors
import torch.nn.functional as F
import torch
import torch.nn as nn

from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from open_clip import create_model, tokenizer
from myutils import UnNormalize
from prompts.imagenet_template import openai_imagenet_template

try:
    from eomt.infer import get_eomt
except:
    print("EoMT is not installed")
try:
    from CropFormer.demo_mask2former.demo import get_entityseg
    from detectron2.data.detection_utils import read_image
except:
    print("EntitySeg is not installed")
try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except:
    print("SAM2 is not installed")

import logging


@MODELS.register_module()
class CorrCLIPSegmentation(BaseSegmentor):
    def __init__(self, clip_type, model_type, dino_type, name_path, device=torch.device('cuda'),
                 prob_thd=0.0, logit_scale=40, slide_stride=112, slide_crop=336, instance_mask_path=None, mask_generator=None):

        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            bgr_to_rgb=True
        )
        super().__init__(data_preprocessor=data_preprocessor)

        self.clip = create_model(model_type, pretrained=clip_type, precision='fp16')
        self.clip.eval().to(device)
        for p in self.clip.parameters():
            p.requires_grad = False
        self.tokenizer = tokenizer.tokenize

        self.dino = torch.hub.load('facebookresearch/dino:main', dino_type)
        self.dino.eval().to(device)
        for p in self.dino.parameters():
            p.requires_grad = False
        self.dino = self.dino.half()

        self.dummy = nn.Linear(1, 1)

        self.unnorm = UnNormalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.generate_category_embeddings(name_path, device)

        self.dtype = self.query_features.dtype
        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.instance_mask_path = instance_mask_path
        self.device = device

        self.set_mask_generator(mask_generator)

    @torch.inference_mode()
    def forward_feature(self, img, masks):
        if type(img) == list:
            img = img[0]

        imgs_norm = [self.norm(self.unnorm(img[i])) for i in range(len(img))]
        imgs_norm = torch.stack(imgs_norm, dim=0)
        imgs_norm = imgs_norm.half()

        feat_out = {}
        def hook_fn_forward_qkv(module, input, output):
            feat_out["qkv"] = output
        self.dino._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(
            hook_fn_forward_qkv)

        # Forward pass in the model
        feat = self.dino.get_intermediate_layers(imgs_norm, n=1)[-1]

        patch_size = self.dino.patch_embed.patch_size
        feat_shape = (imgs_norm[0].shape[-2] // patch_size, imgs_norm[0].shape[-1] // patch_size)
        nb_im = feat.shape[0]  # Batch size
        nb_tokens = feat.shape[1]  # Number of tokens

        qkv = feat_out["qkv"].reshape(nb_im, nb_tokens, 3, -1).permute(2, 0, 1, 3)
        dino_feats = qkv[0] + qkv[1]  #B, L, C
        dino_feats = dino_feats[:, 1:, ]
        dino_feats = F.normalize(dino_feats, dim=-1)

        image_features = self.clip.encode_image(img.half(), dino_feats=dino_feats, feat_shape=feat_shape, instance_masks=masks)

        image_features = F.normalize(image_features, dim=-1)
        logits = image_features @ self.query_features.T
        logits = logits.permute(0, 2, 1).reshape(-1, logits.shape[-1], *feat_shape)
        logits = nn.functional.interpolate(logits, size=img.shape[-2:], mode='bilinear')

        return logits

    def forward_slide(self, img, instance_masks, img_metas, stride=112, crop_size=224):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = self.num_queries
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_instance_masks = instance_masks[:, :, y1:y2, x1:x2]

                # pad image when (image_size % patch_size != 0)
                H, W = crop_img.shape[2:]  # original image shape
                pad = self.compute_padsize(H, W, 56)

                if any(pad):
                    crop_img = nn.functional.pad(crop_img, pad)  # zero padding
                    crop_instance_masks = nn.functional.pad(crop_instance_masks, pad, value=10000)
                crop_seg_logit = self.forward_feature(crop_img, crop_instance_masks).detach()

                # mask cutting for padded image
                if any(pad):
                    l, t = pad[0], pad[2]
                    crop_seg_logit = crop_seg_logit[:, :, t:t + H, l:l + W]

                preds += nn.functional.pad(crop_seg_logit,
                                           (int(x1), int(preds.shape[3] - x2), int(y1),
                                            int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat
        img_size = img_metas[0]['ori_shape'][:2]
        logits = nn.functional.interpolate(preds, size=img_size, mode='bilinear')

        torch.cuda.empty_cache()

        return logits

    def predict(self, inputs, data_samples):
        if data_samples is None:    # demo_gradio
            inputs, img_path = inputs
            batch_img_metas = [dict(ori_shape=inputs.shape[2:])] * inputs.shape[0]
            instance_masks = (self.generate_mask(img_path)).unsqueeze(0)
        else:   # evaluation
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
            instance_masks = [self.generate_mask(data_sample.img_path) for data_sample in data_samples]
            instance_masks = torch.stack(instance_masks, dim=0)

        self.instance_masks = instance_masks.int()
        instance_masks = F.interpolate(instance_masks.unsqueeze(1).float(), size=inputs.shape[2:], mode='nearest').int()

        seg_logits = self.forward_slide(inputs, instance_masks, batch_img_metas, self.slide_stride, self.slide_crop)

        return self.postprocess_result(seg_logits, data_samples)

    def postprocess_result(self, seg_logits, data_samples):
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            seg_logit = seg_logits[i] * self.logit_scale
            seg_logit = seg_logit.softmax(0)  # n_queries * h * w

            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_logits_background = seg_logit[:num_queries - num_cls + 1]
                seg_logits_background = seg_logits_background.max(0, keepdim=True)[0]
                seg_logits_stuff = seg_logit[num_queries - num_cls + 1:]
                seg_logit = torch.cat([seg_logits_background, seg_logits_stuff])

            seg_pred = seg_logit.argmax(0, keepdim=True)
            seg_pred[seg_logit.max(0, keepdim=True)[0] < self.prob_thd] = 0

            # Map Correction
            mask_values = torch.unique(self.instance_masks[i])
            mask_values = mask_values[1:]
            masks = mask_values.unsqueeze(1).unsqueeze(1) == self.instance_masks[i].unsqueeze(0).expand(len(mask_values), -1, -1)
            masks = masks.unsqueeze(1)
            for mask in masks:
                seg_pred[mask] = torch.mode(seg_pred[mask])[0]

            if data_samples is None:    # demo_gradio
                return seg_pred
            else:   # evaluation
                data_samples[i].set_data({
                    'seg_logit':
                        PixelData(**{'data': seg_logit}),
                    'pred_sem_seg':
                        PixelData(**{'data': seg_pred})
                })
        return data_samples

    def generate_category_embeddings(self, name_path, device=torch.device('cuda')):
        query_words, self.query_idx = get_cls_idx(name_path)
        self.num_queries = len(query_words)
        self.num_classes = max(self.query_idx) + 1
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        query_features = []
        with torch.inference_mode():
            for qw in query_words:
                query = self.tokenizer([temp(qw) for temp in openai_imagenet_template]).to(device)
                feature = self.clip.encode_text(query)
                feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
        self.query_features = torch.cat(query_features, dim=0).detach()

    def set_mask_generator(self, generator_type):
        self.mask_generator_type = generator_type
        if generator_type == 'eomt':
            self.mask_generator = get_eomt(cfg_file="eomt_large_640.yaml", use_compile=True)  # use torch.compile
        elif generator_type == 'mask2former':
            self.processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-coco-panoptic")
            self.mask_generator = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-panoptic", device_map=self.device)
            logging.disable(logging.WARNING)
        elif generator_type == 'entityseg':
            self.confidence_threshold = 0.5
            self.mask_generator = get_entityseg(cfg_file="mask2former_hornet_3x.yaml", ckpt_path="Mask2Former_hornet_3x_576d0b.pth")
        elif generator_type == 'sam2':
            sam2 = build_sam2("sam2_hiera_l.yaml", "sam2_hiera_large.pt", device=self.device, apply_postprocessing=False)
            sam2 = sam2.half()
            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=sam2,
                points_per_side=8,
                pred_iou_thresh=0.4,
                stability_score_thresh=0.4,
                multimask_output=False,
            )

    def generate_mask(self, img_path):
        # output shape: [H, W];
        # type is int and the minimum denotes the union of unsegmented regions

        if self.mask_generator_type is None:
            # load pre-generated SAM2 masks
            instance_mask = np.load(os.path.join(self.instance_mask_path, Path(img_path).stem + '.npz'))['instance_mask']
            instance_mask = torch.from_numpy(instance_mask).to(self.device)
        elif self.mask_generator_type == 'eomt':
            img = tv_tensors.Image(Image.open(img_path).convert("RGB"))
            with torch.inference_mode(), torch.autocast(dtype=torch.float16, device_type="cuda"):
                imgs = [img.to(self.device)]
                img_sizes = [img.shape[-2:] for img in imgs]

                transformed_imgs = self.mask_generator.resize_and_pad_imgs_instance_panoptic(imgs)
                mask_logits_per_layer, class_logits_per_layer = self.mask_generator(transformed_imgs)
                mask_logits = F.interpolate(mask_logits_per_layer[-1], self.mask_generator.img_size, mode="bilinear")
                mask_logits = self.mask_generator.revert_resize_and_pad_logits_instance_panoptic(mask_logits, img_sizes)

                preds = self.mask_generator.to_per_pixel_preds_panoptic(
                    mask_logits,
                    class_logits_per_layer[-1],
                    self.mask_generator.stuff_classes,
                    self.mask_generator.mask_thresh,
                    self.mask_generator.overlap_thresh,
                )[0]

            instance_mask = preds[..., 1]
        elif self.mask_generator_type == 'mask2former':
            image = Image.open(img_path).convert("RGB")
            inputs = self.processor(image, return_tensors="pt").to(self.device)

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                outputs = self.mask_generator(**inputs)

            result = self.processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
            instance_mask = result["segmentation"]
        elif self.mask_generator_type == 'entityseg':
            img = read_image(img_path, format="BGR")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                predictions = self.mask_generator(img)

            pred_masks = predictions["instances"].pred_masks
            pred_scores = predictions["instances"].scores

            selected_indexes = (pred_scores >= self.confidence_threshold)
            selected_scores = pred_scores[selected_indexes]
            selected_masks = pred_masks[selected_indexes]
            _, m_H, m_W = selected_masks.shape
            instance_mask = torch.zeros((m_H, m_W), dtype=torch.int, device=self.device)

            selected_scores, ranks = torch.sort(selected_scores)
            ranks = ranks + 1
            for index in ranks:
                instance_mask[(selected_masks[index - 1] == 1)] = int(index)
        elif self.mask_generator_type == 'sam2':
            image = Image.open(img_path).convert("RGB")
            image = np.array(image)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                masks = self.mask_generator.generate(image)
            instance_mask = np.zeros((image.shape[0], image.shape[1]), dtype=int)
            if len(masks) != 0:
                sorted_anns = sorted(masks, key=(lambda x: x['area']))  # predicted_iou
                instance_id = 1
                for ann in sorted_anns:
                    m = ann['segmentation']
                    instance_mask[m] = instance_id
                    instance_id += 1
            instance_mask = torch.from_numpy(instance_mask).to(self.device)
        else:
            instance_mask = None

        return instance_mask

    def compute_padsize(self, H: int, W: int, patch_size: int):
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l

        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t

        return l, r, t, b

    def _forward(data_samples):
        """
        """

    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """

    def extract_feat(self, inputs):
        """
        """

    def _stack_batch_gt(self, batch_data_samples):
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)

    def loss(self, inputs, data_samples):
        """
        """


def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split('; ')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices
