_base_ = '../base_config.py'

# model settings
model = dict(
    num_support=3,
    name_path='./configs/mess/foodseg103.txt',
    instance_mask_path='/home/zdk/code/corrclip2/datasets/region_masks/foodseg103_entityseg',
    prob_thd=0.1,   # 0.25 OURS   0.1 CLIP
)

# dataset settings
dataset_type = 'FoodSeg103'
data_root = '/home/zdk/code/corrclip2/datasets/FoodSeg103'

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
            img_path='Images/img_dir/test', seg_map_path='Images/ann_dir/test'),
        pipeline=test_pipeline))