_base_ = './base_config.py'

# model settings
model = dict(
    num_support=20,
    name_path='./configs/cls_voc21.txt',
    instance_mask_path='data/instance_mask/voc',
    prob_thd= 0.2
)

# dataset settings
dataset_type = 'PascalVOCDataset'
data_root = 'data/VOC2012'

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
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/Segmentation/val.txt',
        pipeline=test_pipeline))