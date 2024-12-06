# dataset settings
dataset_type = "SMCdataset"
data_root = '/content/drive/MyDrive/SMC/'
test_data_root = '/content/drive/MyDrive/SMC/test/'
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True
)
crop_size = (291, 80)
classes = ('background', 'meniscus')
palette = [[128, 0, 0], [0, 128, 0]]

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", reduce_zero_label=False),
    dict(type="Resize", img_scale=(291, 80), keep_ratio=False),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]

val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(291, 80),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=False),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(291, 80),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=False),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="train/images",
        ann_dir="train/labels",
        pipeline=train_pipeline,
        classes=classes,
        palette=palette,
    ),
    trainval=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=["train/images", "train/images"],
        ann_dir=["train/labels", "train/labels"],
        pipeline=train_pipeline,
        classes=classes,
        palette=palette,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="train/images",
        ann_dir="train/labels",
        pipeline=val_pipeline,
        classes=classes,
        palette=palette,
    ),
    test=dict(
        type=dataset_type,
        data_root=test_data_root,
        img_dir="images",
        ann_dir="labels",
        pipeline=test_pipeline,
        classes=classes,
        palette=palette,
    ),
)
