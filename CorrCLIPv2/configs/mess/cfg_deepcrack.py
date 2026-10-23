_base_ = '../base_config.py'

# model settings
model = dict(
    num_support=3,
    name_path='./configs/mess/deepcrack.txt',
    # instance_mask_path='/home/zdk/code/corrclip2/datasets/region_masks/DeepCrack_entityseg',
    instance_mask_path='/home/zdk/code/corrclip2/datasets/region_masks/DeepCrack_sam_multimask_output_area-descending_thresh-0.7',
    slide_stride=112,
    slide_crop=336,
)

# dataset settings
dataset_type = 'DeepCrackDATASET'
data_root = '/home/zdk/code/corrclip2/datasets/DeepCrack'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 784), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='test_img', seg_map_path='annotations_detectron2/test'),
        pipeline=test_pipeline))