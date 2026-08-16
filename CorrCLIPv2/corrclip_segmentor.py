import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import faiss

import numpy as np
from PIL import Image
from pathlib import Path

from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.registry import MODELS
from mmengine.structures import PixelData

import torch.nn.functional as F
import torch
import torch.nn as nn

import open_clip
from open_clip import create_model
from prompts.imagenet_template import openai_imagenet_template

from mask_generators import build_mask_generator
from vfms import build_vfm

from torch.utils.data import Dataset

MB_ROOT = 'memory_bank'   # fixed memory-bank root; memory_bank_builder writes its outputs here too


@MODELS.register_module()
class CorrCLIPSegmentation(BaseSegmentor):
    def __init__(self, clip_type, model_type, name_path,
                 aux_clip_type, aux_model_type,
                 vfm_type,
                 device=torch.device('cuda'), dtype=torch.float16,
                 prob_thd=0.0, logit_scale=40, slide_stride=112, slide_crop=336, instance_mask_path=None, mask_generator=None,
                 background_id=0, use_support_embedding=True, use_aux_model=True, num_support=20,
                 mb_region_set=None,
                 mb_main_encoder=None, mb_aux_encoder=None):

        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            bgr_to_rgb=True
        )
        super().__init__(data_preprocessor=data_preprocessor)

        self.dtype = dtype
        # Whether to fuse support features retrieved from the memory bank.
        # Off = only the image's own features are used, i.e. CorrCLIP v1.
        self.use_support_embedding = use_support_embedding
        # Whether to fuse the auxiliary CLIP (DFN-B). Off = main CLIP only; aux is never loaded.
        self.use_aux_model = use_aux_model

        self.clip = create_model(model_type, pretrained=clip_type, precision='fp16')
        self.clip.eval().to(device)
        for p in self.clip.parameters():
            p.requires_grad = False
        self.tokenizer = open_clip.get_tokenizer(model_type)

        self.dummy = nn.Linear(1, 1)

        self.query_features = self.generate_category_embeddings(name_path,
                                                                clip=self.clip,
                                                                tokenizer=self.tokenizer,
                                                                device=device)

        if self.use_support_embedding and self.use_aux_model:
            aux_clip = create_model(aux_model_type, pretrained=aux_clip_type, precision='fp16').eval().to(device)
            self.aux_query_features = self.generate_category_embeddings(name_path,
                                                                        clip=aux_clip,
                                                                        tokenizer=open_clip.get_tokenizer(aux_model_type),
                                                                        device=device)
            aux_clip_output_dim = aux_clip.visual.output_dim
            del aux_clip
            torch.cuda.empty_cache()
        else:
            self.aux_query_features = None

        self.model_type = model_type
        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.instance_mask_path = instance_mask_path
        self.device = device
        self.vfm_type = vfm_type
        self.background_id = background_id

        self.set_mask_generator(mask_generator)

        clip_output_dim = self.clip.visual.output_dim

        self.num_support = num_support
        self.alpha = 0.5

        if not self.use_support_embedding:
            # CorrCLIP v1 mode: no memory bank / FAISS index loaded
            self.all_clip_feats = None
            self.all_aux_clip_feats = None
            self.gpu_index = None
        else:
            # === memory bank: in-RAM .pt feature stores + IDMap2 FAISS index (built by memory_bank_builder) ===
            self.all_clip_feats = PTTensorDataset(
                os.path.join(MB_ROOT, 'clip_embeddings', f'{mb_region_set}_{mb_main_encoder}.pt'), clip_output_dim)
            self.all_aux_clip_feats = (PTTensorDataset(
                os.path.join(MB_ROOT, 'clip_embeddings', f'{mb_region_set}_{mb_aux_encoder}.pt'), aux_clip_output_dim)
                if self.use_aux_model else None)

            # Index filename = bank + deployment VFM: the index must be built with the same VFM that produces the queries
            faiss_path = os.path.join(MB_ROOT, 'vfm_embeddings', f'{mb_region_set}_{vfm_type}.faiss')
            # Inner index goes to GPU for search; id_map (internal position -> store row) stays on CPU, applied after search
            cpu_index = faiss.read_index(faiss_path)
            self.mb_id_map = faiss.vector_to_array(cpu_index.id_map).astype('int64')
            inner = faiss.downcast_index(cpu_index.index)
            res = faiss.StandardGpuResources()
            self.gpu_index = faiss.index_cpu_to_gpu(res, int(torch.cuda.current_device()), inner)
            if hasattr(inner, 'nlist'):                # only IVFPQ has nprobe; small Flat indexes don't
                self.gpu_index.nprobe = min(256, inner.nlist)
            print(f"[memory_bank] {mb_region_set}: {len(self.all_clip_feats)} features, FAISS {faiss_path} ({cpu_index.ntotal} vectors)")
        torch.cuda.empty_cache()

        # VFM (vfms/ factory): radiov3 / radiov2.5 / dinov2 / dinov3 / pe.
        # Drives both the correlation matrix and the retrieval queries; must match the
        # VFM that built the bank index {bank}_{vfm_type}.faiss.
        self.vfm = build_vfm(vfm_type, device)

    @torch.inference_mode()
    def forward_feature(self, img, masks, vfm_feats):
        if type(img) == list:
            img = img[0]

        h, w = img.shape[2:]
        feat_shape = h // self.clip.visual.patch_size[0] * 2, w // self.clip.visual.patch_size[1] * 2
        vfm_feats = F.interpolate(vfm_feats, size=feat_shape, mode='bilinear')
        vfm_feats = vfm_feats.flatten(2, 3).transpose(1, 2)
        vfm_feats = F.normalize(vfm_feats, dim=-1)

        # Forward pass in the model
        image_features = self.clip.encode_image(img.half(), dino_feats=vfm_feats, feat_shape=feat_shape, instance_masks=masks)

        image_features = F.normalize(image_features, dim=-1)
        logits = image_features @ self.query_features.T
        logits = logits.permute(0, 2, 1).reshape(-1, logits.shape[-1], *feat_shape)
        logits = nn.functional.interpolate(logits, size=img.shape[-2:], mode='bilinear')

        return logits

    @torch.inference_mode()
    def forward_slide(self, img, instance_masks, img_metas, stride=112, crop_size=224):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        raw_img = Image.open(img_metas[0]['img_path']).convert("RGB")
        vfm_feats = self.vfm.extract_dense(raw_img)[0].to(self.dtype)

        img_size = img.shape[2:]
        scale_instance_masks = instance_masks[0]
        vfm_feats = F.interpolate(vfm_feats, size=img_size, mode='bilinear')
        vfm_feats = F.normalize(vfm_feats, dim=1)

        support_clip_logits = None
        if self.use_support_embedding:
            # boolean mask per region (used to scatter support logits back onto the image)
            mask_values = torch.unique(scale_instance_masks[0])
            support_mask = [(v == scale_instance_masks[0]) for v in mask_values]

            # VFM region features serve as retrieval queries; value reconstruction also uses vfm_feats
            index_feats_nhwc = vfm_feats.permute(0, 2, 3, 1)
            query_embeddings = torch.stack([index_feats_nhwc[0][m].mean(0) for m in support_mask])
            query_embeddings = F.normalize(query_embeddings, dim=-1)
            query_np = query_embeddings.float().cpu().numpy()
            distances, indices = self._faiss_search(query_np, self.num_support)

            support_clip_logits = self.get_support_clip_logits(indices, support_mask, img_size, self.query_features, self.all_clip_feats).half()
            if self.use_aux_model:
                support_clip_logits.add_(self.get_support_clip_logits(indices, support_mask, img_size, self.aux_query_features, self.all_aux_clip_feats).half())
                support_clip_logits.mul_(0.5)

        torch.cuda.empty_cache()

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
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img)).half()
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img)).half()
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
                crop_vfm_feats= vfm_feats[:, :, y1:y2, x1:x2]

                # pad image when (image_size % patch_size != 0)
                H, W = crop_img.shape[2:]  # original image shape
                pad = self.compute_padsize(H, W, 56)

                if any(pad):
                    crop_img = F.pad(crop_img, pad)  # zero padding
                    crop_instance_masks = F.pad(crop_instance_masks, pad, value=10000)
                    crop_vfm_feats = nn.functional.pad(crop_vfm_feats, pad, mode='replicate')
                crop_seg_logit = self.forward_feature(crop_img, crop_instance_masks, crop_vfm_feats).detach().half()

                # mask cutting for padded image
                if any(pad):
                    l, t = pad[0], pad[2]
                    crop_seg_logit = crop_seg_logit[:, :, t:t + H, l:l + W]

                preds += F.pad(crop_seg_logit,
                                           (int(x1), int(preds.shape[3] - x2), int(y1),
                                            int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat

        if not self.use_support_embedding:
            # CorrCLIP v1: image's own features only, no fusion with retrieved support_clip_logits
            preds = F.interpolate(preds, size=img_metas[0]['ori_shape'][:2], mode='bilinear')
        else:
            preds.mul_(self.alpha)  # in-place, no temporary
            preds.add_(support_clip_logits * (1 - self.alpha))  # only the support*(1-alpha) temporary; lower peak memory
            preds = F.interpolate(preds, size=img_metas[0]['ori_shape'][:2], mode='bilinear')

        return preds

    def _faiss_search(self, query_np, k):
        """FAISS search returning (distances, indices = store row ids).
        The GPU searches the inner index for internal positions; id_map then
        translates them to store rows."""
        D, P = self.gpu_index.search(query_np, k)
        n = self.mb_id_map.shape[0]
        I = np.where(P >= 0, self.mb_id_map[np.clip(P, 0, n - 1)], -1)   # internal position -> store row
        return D, I

    def get_support_clip_logits(self, indices, support_mask, img_size, query_features, all_clip_feats):
        support_clip_embeddings = []
        for index in indices:
            index = [i for i in index if i != -1]
            batch_cpu = all_clip_feats[index]
            batch_gpu = batch_cpu.to('cuda', non_blocking=True)
            support_clip_embeddings.append(batch_gpu.mean(0))
        support_clip_embeddings = torch.stack(support_clip_embeddings)
        support_clip_embeddings = F.normalize(support_clip_embeddings, dim=-1)
        support_clip_logits = support_clip_embeddings @ query_features.T
        support_mask = torch.stack(support_mask).flatten(1, -1).to(support_clip_logits.dtype)
        support_clip_logits = support_mask.T @ support_clip_logits
        support_clip_logits = support_clip_logits.unflatten(0, img_size)
        support_clip_logits = support_clip_logits.permute(2, 0, 1).unsqueeze(0)

        return support_clip_logits

    def predict(self, inputs, data_samples):
        batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        img_paths = [data_sample.img_path for data_sample in data_samples]
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
            H, W = seg_logit.shape[1], seg_logit.shape[2]

            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)

            if num_cls != num_queries:
                out = seg_logit.new_full((num_cls, H, W), float('-inf'))

                # scatter_reduce_ takes the per-class max without a large intermediate tensor
                out.scatter_reduce_(
                    0,  # reduce over the class dimension
                    self.query_idx.view(num_queries, 1, 1).expand(-1, H, W),
                    seg_logit,
                    reduce="amax",
                    include_self=True
                )

                seg_logit = out

            seg_pred = seg_logit.argmax(0, keepdim=True)
            seg_pred[seg_logit.max(0, keepdim=True)[0] < self.prob_thd] = self.background_id

            # Map Correction
            mask_values = torch.unique(self.instance_masks[i])
            mask_values = mask_values[1:]
            # masks = mask_values.unsqueeze(1).unsqueeze(1) == self.instance_masks[i].unsqueeze(0).expand(len(mask_values), -1, -1)
            # masks = masks.unsqueeze(1)
            # for mask in masks:
            #     seg_pred[mask] = torch.mode(seg_pred[mask])[0]
            for value in mask_values:
                mask = value == self.instance_masks[i].unsqueeze(0)
                value, _ = torch.mode(seg_pred[mask])
                seg_pred[mask] = value

            data_samples[i].set_data({
                'seg_logit':
                    PixelData(**{'data': seg_logit}),
                'pred_sem_seg':
                    PixelData(**{'data': seg_pred})
            })
        return data_samples

    def generate_category_embeddings(self, name_path, clip, tokenizer, device=torch.device('cuda')):
        query_words, self.query_idx = get_cls_idx(name_path)
        self.num_queries = len(query_words)
        self.num_classes = max(self.query_idx) + 1
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        query_features = []
        with torch.inference_mode():
            for qw in query_words:
                query = tokenizer([temp(qw) for temp in openai_imagenet_template]).to(device)
                feature = clip.encode_text(query)
                feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))

        return torch.cat(query_features, dim=0).detach()

    def set_mask_generator(self, generator_type):
        """generator_type: entityseg / eomt / sam / sam2 (on the fly, see mask_generators/),
        or None (load pre-generated masks from instance_mask_path)."""
        self.mask_generator_type = generator_type
        if generator_type is None:
            self.mask_generator = None
            return
        # deployment hyper-parameters (bank-building ones live in memory_bank_builder/generate_mask.py)
        deploy_kwargs = {
            'entityseg': dict(autocast_dtype=self.dtype),
            'eomt': dict(autocast_dtype=self.dtype),
            'sam': dict(autocast_dtype=self.dtype),
            'sam2': dict(autocast_dtype=self.dtype, model_dtype=self.dtype,
                         points_per_side=8, pred_iou_thresh=0.4,
                         stability_score_thresh=0.4, multimask_output=False),
        }[generator_type]
        self.mask_generator = build_mask_generator(generator_type, self.device, **deploy_kwargs)

    def generate_mask(self, img_path):
        # output shape: [H, W];
        # type is int and the minimum denotes the union of unsegmented regions
        if self.mask_generator_type is None:
            # load pre-generated masks
            instance_mask = np.load(os.path.join(self.instance_mask_path, Path(img_path).stem + '.npz'))['instance_mask'].astype(int)
            return torch.from_numpy(instance_mask).to(self.device)
        return self.mask_generator.generate(img_path)

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


class PTTensorDataset(Dataset):
    """In-RAM feature store: loads {BASE}.pt ([N, D] fp16 tensor, row index = manifest id)
    entirely into RAM at init; lookups are plain tensor fancy-indexing, no disk IO.
    The .pt files are produced by memory_bank_builder (legacy LMDB stores migrate
    bit-identically via memory_bank_builder/lmdb_to_pt.py)."""
    def __init__(self, pt_path, embedding_dim, dtype=torch.float16):
        self.data = torch.load(pt_path, map_location='cpu')
        assert self.data.dim() == 2 and self.data.shape[1] == embedding_dim, \
            f'{pt_path}: shape {tuple(self.data.shape)} does not match embedding_dim={embedding_dim}'
        assert self.data.dtype == dtype, f'{pt_path}: dtype {self.data.dtype} != {dtype}'
        print(f'[PTTensorDataset] {pt_path} -> RAM {tuple(self.data.shape)} {self.data.dtype}')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return self.data[int(idx)]
        if isinstance(idx, torch.Tensor):
            return self.data[idx.to(torch.long)]
        return self.data[torch.as_tensor(idx, dtype=torch.long)]


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
