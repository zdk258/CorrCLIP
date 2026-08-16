import os
import sys
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms, tv_tensors
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch
from open_clip import create_model, tokenizer
from myutils import UnNormalize
from prompts.imagenet_template import openai_imagenet_template
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class CorrCLIPInfer():
    def __init__(self, clip_type, model_type, dino_type, name_path, device=torch.device('cuda'),
                 prob_thd=0.0, logit_scale=40, slide_stride=112, slide_crop=336, instance_mask_path=None, mask_generator=None):
        super().__init__()

        self.clip = create_model(model_type, pretrained=clip_type)
        self.clip.eval().to(device).half()
        for p in self.clip.parameters():
            p.requires_grad = False
        self.tokenizer = tokenizer.tokenize

        self.dino = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8', weights_only=False)
        self.dino.eval().to(device)
        for p in self.dino.parameters():
            p.requires_grad = False
        self.dino = self.dino.half()

        self.dino_qkv_output = None
        self.dino.blocks[-1].attn.qkv.register_forward_hook(self._hook_fn_forward_qkv)

        self.dummy = nn.Linear(1, 1)

        self.unnorm = UnNormalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.instance_mask_path = instance_mask_path
        self.device = device

        self.set_mask_generator(mask_generator)

    def _hook_fn_forward_qkv(self, module, input, output):
        self.dino_qkv_output = output

    @torch.inference_mode()
    def forward_feature(self, img, masks):
        if type(img) == list:
            img = img[0]

        imgs_norm = [self.norm(self.unnorm(img[i])) for i in range(len(img))]
        imgs_norm = torch.stack(imgs_norm, dim=0)
        imgs_norm = imgs_norm.half()

        # Forward pass in the model
        self.dino_qkv_output = None
        feat = self.dino.get_intermediate_layers(imgs_norm, n=1)[-1]

        patch_size = self.dino.patch_embed.patch_size
        feat_shape = (imgs_norm[0].shape[-2] // patch_size, imgs_norm[0].shape[-1] // patch_size)
        nb_im = feat.shape[0]  # Batch size
        nb_tokens = feat.shape[1]  # Number of tokens

        qkv = self.dino_qkv_output.reshape(nb_im, nb_tokens, 3, -1).permute(2, 0, 1, 3)
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

        if torch.cuda.is_available():
          torch.cuda.empty_cache()

        return logits

    def predict(self, inputs, data_samples):
        inputs, img_path = inputs
        batch_img_metas = [dict(ori_shape=inputs.shape[2:])] * inputs.shape[0]
        instance_masks = (self.generate_mask(img_path)).unsqueeze(0)

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

            return seg_pred

    def generate_category_embeddings(self, name_path):
        device=self.device
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
        sam2 = build_sam2("sam2_hiera_l.yaml", "sam2_hiera_large.pt", device=self.device, apply_postprocessing=False)
        self.sam2 = sam2.half()
        self.seg_sam2_params()

    def seg_sam2_params(self, points_per_side=16, pred_iou_thresh=0.4, stability_score_thresh=0.4, multimask_output=False):
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.sam2,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            multimask_output=multimask_output,
        )

    def generate_mask(self, img_path):
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.float16):
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


import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from scipy.ndimage import label
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
import warnings
import sys

# --- Auxiliary functions ---
def generate_distinct_colors(n):
    # Generate vibrant colors for different categories
    hues = np.linspace(0, 1, n, endpoint=False)
    sats = np.ones(n) * 0.8; vals = np.ones(n) * 0.9
    hsv_colors = np.stack((hues, sats, vals), axis=-1)
    return (hsv_to_rgb(hsv_colors) * 255).astype(np.uint8)


# --- Core Processing Function ---
def run_segmentation(image_path, input_text, model, device):
    """
    Receives input and performs the full segmentation and visualization process.
    """

    # Read image and update categories
    input_img = Image.open(image_path).convert("RGB")
    with open('./configs/my_name.txt', 'r') as file:
        lines = file.readlines()
    name_list = [line.strip() for line in lines]

    if set(name_list) != set(list(dict.fromkeys(item.strip() for item in input_text.split(',')))):
        name_list = list(dict.fromkeys(item.strip() for item in input_text.split(',')))
        with open('./configs/my_name.txt', 'w') as writers:
            for i in range(len(name_list)):
                if i == len(name_list) - 1:
                    writers.write(name_list[i])
                else:
                    writers.write(name_list[i] + '\n')
        writers.close()
        model.generate_category_embeddings('./configs/my_name.txt')

    class_names = {index: value for index, value in enumerate(name_list)}

    # Image preprocessing and model prediction
    print("  - Running model prediction...")
    img_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    ])(input_img)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    seg_pred = model.predict([img_tensor, image_path], data_samples=None)
    seg_pred = seg_pred.data.cpu().numpy().squeeze()

    regions = {}
    for label_id in np.unique(seg_pred):
        labeled_array, _ = label(seg_pred == label_id)
        sizes = np.bincount(labeled_array.ravel())
        max_idx = np.argmax(sizes[1:]) + 1

        region = (labeled_array == max_idx)
        y, x = np.where(region)
        center_y, center_x = int(np.mean(y)), int(np.mean(x))

        if y[0] + 10 < region.shape[0] and x[0] + 10 < region.shape[1]:
            regions[label_id] = (center_y, x[0] + 10)
        else:
            regions[label_id] = (center_y, x[0])
    output_str = '; '.join(f"{class_names[idx]}" for idx, coor in regions.items())

    palette = generate_distinct_colors(len(name_list))
    seg_pred_color = palette[seg_pred]

    seg_pred_color = (seg_pred_color * 0.7 + np.array(input_img) * 0.3).astype(np.uint8)

    seg_pred_color = Image.fromarray(seg_pred_color)
    draw = ImageDraw.Draw(seg_pred_color)
    font = ImageFont.load_default(size=max(seg_pred.shape[0], seg_pred.shape[1]) // 50)
    for class_id, (center_y, center_x) in regions.items():
        if class_id in class_names:
            text = class_names[class_id]
            draw.text((center_x, center_y), text, font=font, fill="white", anchor="lt")

    print("\n✅ Processing complete!")
    return input_img, seg_pred_color, output_str

def show_result(original_image, segmented_image, detected_classes):
    plt.figure(figsize=(18, 9))

    plt.subplot(1, 2, 1)
    plt.title("Original Image", fontsize=16)
    plt.imshow(original_image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Segmentation Result", fontsize=16)
    plt.imshow(segmented_image)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 40)
    print(f"📄 Detected Classes: {detected_classes}")
    print("=" * 40)