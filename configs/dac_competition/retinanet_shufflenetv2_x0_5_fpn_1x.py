# model settings
model = dict(
    type='RetinaNet',
    pretrained='/home/xiongfeng/basemodels/shufflenetv2/shufflenetv2_x0.5.pth',
    backbone=dict(
        type='ShuffleNetV2',
        stages_repeats=[4, 8, 4],
        stages_out_channels=[24, 48, 96, 192],
        frozen_stages=1),
    neck=dict(
        type='FPN',
        in_channels=[24, 48, 96, 192],
        out_channels=64,
        start_level=1,
        add_extra_convs=False,
        num_outs=3),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=2,
        in_channels=64,
        stacked_convs=2,
        feat_channels=64,
        octave_base_scale=4,
        scales_per_octave=[2 ** (-3), 2 ** (-2), 2 ** (-1), 1],
        anchor_ratios=[1.0, 2.0, 3.0, 4.0],
        anchor_strides=[8, 16, 32],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=1)
# dataset settings
dataset_type = 'DacDataset'
data_root = '/home/xiongfeng/dataset/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'dac_train.json',
        img_prefix='',
        img_scale=(640, 360),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=False,
        with_label=True),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'dac_val.json',
        img_prefix='',
        img_scale=(640, 360),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'dac_val.json',
        img_prefix='',
        img_scale=(640, 360),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/home/xiongfeng/work_dirs/dac_competition/retinanet_shufflenetv2_x0_5_fpn_1x'
load_from = None
resume_from = None
workflow = [('train', 1)]
