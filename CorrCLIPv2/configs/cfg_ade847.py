_base_ = './base_config.py'

# model settings
model = dict(
    num_support=20,
    name_path='./configs/cls_ade20k847.txt',
    instance_mask_path='data/instance_mask/ade847'
)

# dataset settings
dataset_type = 'ADE20K847Dataset'
data_root = 'data/ADE20K_2021_17_01'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 448), keep_ratio=True),
    dict(type='MyLoadAnnotations', reduce_zero_label=False),
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
            img_path='images_detectron2/validation',
            seg_map_path='annotations_detectron2/validation'),
        # ann_file='validation.txt',
        pipeline=test_pipeline))