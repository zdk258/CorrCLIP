"""VFM factory: loads a vision foundation model and extracts dense feature maps.

Memory-bank index building (memory_bank_builder/generate_vfm_embeddings.py) and
deployment inference (correlation matrix + retrieval queries in corrclip_segmentor.py)
share the models and preprocessing defined here, so index and queries always live
in the same feature space.
"""
import os

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor

_DIR = os.path.dirname(os.path.abspath(__file__))

# short name -> loading args; dim = feature dimension of the .pt store and FAISS index
VFM_REGISTRY = {
    'radiov3':   dict(kind='radio',  version='c-radio_v3-l',         dim=1024),
    'radiov2.5': dict(kind='radio',  version='radio_v2.5-l',         dim=1024),
    'dinov2':    dict(kind='dinov2', entrypoint='dinov2_vitl14_reg', dim=1024),
    'dinov3':    dict(kind='dinov3', entrypoint='dinov3_vitl16',     dim=1024),
    'pe':        dict(kind='pe',     config='PE-Spatial-L14-448',    dim=1024),
}


class VFMExtractor:
    """Uniform interface: extract_dense(PIL.Image) -> (feats [1,C,h,w] fp16, ref_size).

    ref_size is the size features should be interpolated back to for mask
    alignment: the original image size for radio/dinov2/dinov3, the preprocessed
    tensor size for pe.
    """

    def __init__(self, name, device):
        if name not in VFM_REGISTRY:
            raise ValueError(f"Unsupported VFM: {name} (choices: {', '.join(VFM_REGISTRY)})")
        meta = VFM_REGISTRY[name]
        self.name = name
        self.kind = meta['kind']
        self.dim = meta['dim']
        self.device = device

        if self.kind == 'radio':
            self.model = torch.hub.load('NVlabs/RADIO', 'radio_model', version=meta['version'],
                                        progress=True, trust_repo=True)
        elif self.kind == 'dinov2':
            self.model = torch.hub.load('facebookresearch/dinov2', meta['entrypoint'],
                                        trust_repo=True)
        elif self.kind == 'dinov3':
            self.model = torch.hub.load(os.path.join(_DIR, 'dinov3'), meta['entrypoint'],
                                        source='local')
        elif self.kind == 'pe':
            from pe.core.vision_encoder import pe as pe_module  # vendored: vfms/pe
            ckpt = f"{meta['config']}.pt"  # weight file at repo root; auto-downloaded if absent
            self.model = pe_module.VisionTransformer.from_config(
                meta['config'], pretrained=True,
                checkpoint_path=ckpt if os.path.exists(ckpt) else None)
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad = False

        if self.kind in ('dinov2', 'dinov3'):
            self.norm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        elif self.kind == 'pe':
            import pe.core.vision_encoder.transforms as pe_transforms
            self.preprocess = pe_transforms.get_image_transform(self.model.image_size)
            self.preprocess.transforms = self.preprocess.transforms[1:]  # drop the squash-resize: run at native resolution

    @torch.inference_mode()
    def extract_dense(self, img):
        """img: PIL.Image (RGB). Returns (feats [1,C,h,w] fp16 on device, ref_size)."""
        if self.kind == 'radio':
            img_tensor = pil_to_tensor(img).to(dtype=torch.float32, device=self.device)
            img_tensor.div_(255.0)  # RADIO expects inputs in [0, 1]
            img_tensor = img_tensor.unsqueeze(0)
            ref_size = img_tensor.shape[-2:]
            nearest_res = self.model.get_nearest_supported_resolution(*ref_size)
            img_tensor = F.interpolate(img_tensor, nearest_res, mode='bilinear', align_corners=False)
            with torch.autocast('cuda', dtype=torch.float16):
                feats = self.model(img_tensor, feature_fmt='NCHW')[1].half().contiguous()
        elif self.kind in ('dinov2', 'dinov3'):
            img_tensor = self.norm(img).unsqueeze(0).to(self.device, non_blocking=True)
            ref_size = img_tensor.shape[-2:]
            # DINO backbones need side lengths divisible by the patch size; resize to the nearest multiple
            patch = getattr(self.model, 'patch_size', 14)
            H, W = ref_size
            newH = max(patch, int(round(H / patch)) * patch)
            newW = max(patch, int(round(W / patch)) * patch)
            if (newH, newW) != (H, W):
                img_tensor = F.interpolate(img_tensor, size=(newH, newW), mode='bilinear', align_corners=False)
            with torch.autocast('cuda', dtype=torch.float16):
                feats = self.model.get_intermediate_layers(img_tensor, reshape=True)[0].half().contiguous()
        elif self.kind == 'pe':
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device, non_blocking=True)
            ref_size = img_tensor.shape[-2:]
            with torch.autocast('cuda', dtype=torch.float16):
                feats = self.model.forward_features(img_tensor, strip_cls_token=True).half().contiguous()
            patch = self.model.patch_size
            feats = feats.unflatten(dim=1, sizes=(ref_size[0] // patch, ref_size[1] // patch))
            feats = feats.permute(0, 3, 1, 2).contiguous()
        return feats, ref_size


def build_vfm(name, device):
    return VFMExtractor(name, device)
