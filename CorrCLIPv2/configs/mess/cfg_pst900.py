_base_ = '../base_config.py'

# model settings
model = dict(
    num_support=3,
    name_path='./configs/mess/pst900.txt',
    instance_mask_path='/home/zdk/code/corrclip2/datasets/region_masks/PST900_entityseg',
    slide_stride=112,
    slide_crop=224,
)

# dataset settings
dataset_type = 'PST900DATASET'
data_root = '/home/zdk/code/corrclip2/datasets/PST900_RGBT_Dataset'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 672), keep_ratio=True),
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
            img_path='test/thermal', seg_map_path='test/labels'),
        pipeline=test_pipeline))