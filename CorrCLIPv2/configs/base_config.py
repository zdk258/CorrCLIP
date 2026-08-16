# base configurations

# CLIP backbone scale: 'large' (default, paper Table 2) / 'base' (Table 3: all MESS methods use base).
# MESS configs already override to base via configs/mess/base_mess.py — no need to change this here.
_clip_scale = 'large'
_model_type = {'base': 'ViT-B-16-quickgelu', 'large': 'ViT-L-14-quickgelu'}[_clip_scale]

# ===== memory bank (in-RAM .pt feature stores + IDMap2 FAISS index, built by memory_bank_builder/build.py) =====
# Bank: 'coco' (default, paper main results) / 'monet' (Table 2 last row, fully synthetic disjoint bank)
# / 'gpic' / 'sa1b', or any custom bank name (build.py --data_name). Files live at
#   memory_bank/clip_embeddings/{bank}_{encoder}.pt and memory_bank/vfm_embeddings/{bank}_{vfm}.faiss
_mb_bank = 'coco'
_mb_main_encoder = {'base': 'metaclip_b', 'large': 'metaclip_l'}[_clip_scale]

model = dict(
    type='CorrCLIPSegmentation',

    # ========== 1) CorrCLIP v1: single-image open-vocabulary segmentation ==========
    # Instance mask source: 'entityseg' / 'eomt' / 'sam' / 'sam2' (generated on the fly,
    # see mask_generators/), or None = load pre-generated masks from the dataset config's
    # instance_mask_path
    mask_generator=None,
    # CLIP backbone
    clip_type='metaclip_fullcc',
    model_type=_model_type,
    # Vision foundation model: 'radiov3' (default) / 'radiov2.5' / 'dinov2' / 'dinov3' / 'pe'
    # (see vfms/). Provides the correlation matrix; with the memory bank enabled it also
    # selects the retrieval index {bank}_{vfm_type}.faiss (build it first via build.py --vfm)
    vfm_type='radiov3',

    # ========== 2) CorrCLIPv2: main memory bank (support retrieval) ==========
    # True  -> fuse support features retrieved from the memory bank (CorrCLIPv2);
    # False -> use only the image's own features, i.e. CorrCLIP v1 (no bank files needed).
    use_support_embedding=True,
    mb_region_set=_mb_bank,            # bank name
    mb_main_encoder=_mb_main_encoder,  # main store encoder: {bank}_{mb_main_encoder}.pt (matches CLIP backbone scale)

    # ========== 3) CorrCLIPv2: auxiliary memory bank (DFN-B) ==========
    # True  -> average the support logits of the main and auxiliary banks;
    # False -> main bank only (the no-DFN row of Table 2).
    use_aux_model=True,
    aux_clip_type=None,
    aux_model_type='hf-hub:apple/DFN2B-CLIP-ViT-B-16',
    mb_aux_encoder='dfnclip_b',        # auxiliary store encoder: {bank}_{mb_aux_encoder}.pt
)

test_evaluator = dict(
    type='BoundaryIoUMetric',
    iou_metrics=['mIoU'],
    boundary_dilation_ratio=0.02,
)

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, alpha=1.0, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', interval=5))
