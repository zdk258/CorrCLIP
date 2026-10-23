_base_ = '../base_config.py'

# model settings
model = dict(
    num_support=3,
    name_path='./configs/mess/floodnet.txt',
    instance_mask_path='/home/zdk/code/corrclip2/datasets/region_masks/floodnet_entityseg',
    slide_stride=112,
    slide_crop=336,
    prob_thd=0.3
)

# dataset settings
dataset_type = 'FLOODNETDATASET'
data_root = '/home/zdk/code/corrclip2/datasets/floodnet'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 448), keep_ratio=True),
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
            img_path='images_detectron2/test', seg_map_path='annotations_detectron2/test'),
        pipeline=test_pipeline))