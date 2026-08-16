_base_ = './base_config.py'

# model settings
model = dict(
    num_support=20,
    name_path='./configs/cls_coco_object.txt',
    instance_mask_path='data/instance_mask/coco',
    prob_thd=0.25,
)

# dataset settings
dataset_type = 'COCOObjectDataset'
data_root = 'data/coco'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 336), keep_ratio=True),
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
        reduce_zero_label=False,
        data_prefix=dict(
            img_path='images/val2017', seg_map_path='annotations/val2017'),
        pipeline=test_pipeline))