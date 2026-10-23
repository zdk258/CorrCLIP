_base_ = '../base_config.py'

# model settings
model = dict(
    num_support=3,
    name_path='./configs/mess/zerowaste.txt',
    instance_mask_path='/home/zdk/code/corrclip2/datasets/region_masks/zerowaste_entityseg',
    prob_thd=0.4,
)

# dataset ZeroWasteDATASET
dataset_type = 'ZeroWasteDATASET'
data_root = '/home/zdk/code/corrclip2/datasets/zerowaste-f'

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
            img_path='splits_final_deblurred/test/data', seg_map_path='splits_final_deblurred/test/sem_seg'),
        pipeline=test_pipeline))