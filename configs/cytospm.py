# CytoSPM (ConvNeXt-T) on the PentaCyto base split.
#
# Stage I : class-agnostic visual features (ConvNeXt-T backbone, neck, head).
# Stage II: class-aware structural prompt matching with frozen BioMedCLIP
#           embeddings of the class names and of the cytomorphology prompts.
#
# Train with a global batch of 64 (32 x 2 GPUs); the paper reports epoch 9.
import json as _json
import os as _os

custom_imports = dict(imports=['cytospm'], allow_failed_imports=False)
default_scope = 'mmdet'

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
data_root = _os.environ.get('CYTOSPM_DATA_ROOT', 'data/PentaCyto').rstrip('/') + '/'
train_ann_file = 'annotations/instances_train_dev_disjoint_dev30.json'
val_ann_file = 'annotations/instances_val_dev_disjoint_dev30.json'
class_text_path = 'data/texts/tct_ngc_fullnames_30.json'
text_embed_path = 'data/texts/tct_ngc_fullnames_30_embeddings_biomedclip.pth'
prompt_bank_path = 'data/promptbank/tct_ngc_descriptors_30_structmatch.pt'
organ_mask_path = 'data/texts/tct_ngc_class_organ_mask_base30.pt'
taxonomy_path = 'data/texts/tct_ngc_taxonomy.json'
# Non-diagnostic classes: trained on, but excluded from the base metric.
negative_class_names = _json.load(
    open('data/texts/tct_ngc_dev30_negatives.json', encoding='utf-8'))['negative_class_names']

classes = (
    'respiratory tract-Neutrophil',
    'respiratory tract-Alveolar macrophages',
    'respiratory tract-Ciliated columnar epithelial cells',
    'respiratory tract-Lymphocyte',
    'respiratory tract-Impurity',
    'respiratory tract-Squamous epithelial cells',
    'respiratory tract-Diseased cells',
    'Serous effusion-Negative samples',
    'Serous effusion-Diseased cells',
    'Thyroid gland-PTC',
    'Thyroid gland-SPTC',
    'Thyroid gland-NS',
    'Thyroid gland-Macrophages',
    'Thyroid gland-AUC',
    'Thyroid gland-Negative samples',
    'Thyroid gland-FC',
    'Urine-NHGUC',
    'Urine-SHGUC',
    'Urine-AUC',
    'Urine-HGUC',
    'TCT_CCD-normal',
    'TCT_CCD-ascus',
    'TCT_CCD-asch',
    'TCT_CCD-lsil',
    'TCT_CCD-hsil_scc_omn',
    'TCT_CCD-agc_adenocarcinoma_em',
    'TCT_CCD-vaginalis',
    'TCT_CCD-monilia',
    'TCT_CCD-dysbacteriosis_herpes_act',
    'TCT_CCD-ec',
)
num_classes = len(classes)
img_scale = (640, 640)
meta_keys = ('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
             'pad_param', 'texts', 'organ_id', 'organ_name')

# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
model = dict(
    type='YOLOWorldDetector',
    num_train_classes=num_classes,
    data_preprocessor=dict(
        type='YOLOWDetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    backbone=dict(
        type='MultiModalYOLOBackbone',
        image_model=dict(type='ConvNextVisionBackbone', model_name='tiny'),
        text_model=dict(
            type='PseudoLanguageBackbone', text_embed_path=text_embed_path)),
    neck=dict(type='CSPRepBiFPANNeck', model_size='tiny'),
    bbox_head=dict(
        type='YOLOWorldHead',
        organ_loss_mask=True,
        organ_mask_path=organ_mask_path,
        head_module=dict(
            type='YOLOWorldHeadModule',
            num_classes=num_classes,
            in_channels=[96, 192, 384],
            embed_dims=512,
            model_size='tiny',
            prompt_matching=dict(
                prompt_bank_path=prompt_bank_path,
                top_p=0.01,          # Top-P: fraction of regions per level
                top_q=0.95,          # Top-Q: fraction of prompts per class
                use_prototype=True,  # structural prototype term
                lambda1=0.1,         # weight of the structural prototype term
                lambda2=0.05,        # weight of the fine-grained attribute term
                tau=0.1,
                chunk=8,
                warmup_iters=3180)),  # 2 epochs at a global batch of 64
        prior_generator=dict(
            type='MlvlPointGenerator', offset=0.5, strides=[8, 16, 32]),
        bbox_coder=dict(type='CytoSPMDistancePointBBoxCoder'),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmyoloIoULoss',
            iou_mode='ciou',
            bbox_format='xyxy',
            reduction='sum',
            loss_weight=5.0,
            return_iou=False),
        loss_dfl=dict(
            type='DistributionFocalLoss', reduction='mean', loss_weight=0.375)),
    train_cfg=dict(
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=num_classes,
            use_ciou=True,
            topk=10,
            alpha=0.5,
            beta=2.0,
            eps=1e-9)),
    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.7),
        max_per_img=300))

# ----------------------------------------------------------------------------
# Pipelines and dataloaders
# ----------------------------------------------------------------------------
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10),
    dict(type='CytoSPMKeepRatioResize', scale=img_scale),
    dict(
        type='CytoSPMLetterResize',
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='LoadText'),
    dict(type='OrganExtractor', taxonomy_path=taxonomy_path, strict=True),
    dict(type='PackDetInputs', meta_keys=meta_keys),
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='CytoSPMKeepRatioResize', scale=img_scale),
    dict(
        type='CytoSPMLetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(type='LoadText'),
    dict(type='OrganExtractor', taxonomy_path=taxonomy_path, strict=True),
    dict(type='PackDetInputs', meta_keys=meta_keys),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='yolow_collate'),
    dataset=dict(
        type='MultiModalDataset',
        dataset=dict(
            type='WeCocoDataset',
            data_root=data_root,
            ann_file=train_ann_file,
            data_prefix=dict(img='images/'),
            metainfo=dict(classes=classes),
            filter_cfg=None,
            test_mode=False),
        class_text_path=class_text_path,
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MultiModalDataset',
        dataset=dict(
            type='WeCocoDataset',
            data_root=data_root,
            ann_file=val_ann_file,
            data_prefix=dict(img='images/'),
            metainfo=dict(classes=classes),
            batch_shapes_cfg=None,
            test_mode=True),
        class_text_path=class_text_path,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

# Base metric: COCO AP averaged over organs, each organ evaluated on its own
# classes, with the negative classes excluded.
val_evaluator = dict(
    type='OrganRestrictedCocoMetric',
    ann_file=data_root + val_ann_file,
    metric='bbox',
    classwise=False,
    organ_mask_path=organ_mask_path,
    exclude_class_names=negative_class_names)
test_evaluator = val_evaluator

# ----------------------------------------------------------------------------
# Schedule
# ----------------------------------------------------------------------------
max_epochs = 12
base_lr = 3e-4
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=999)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=base_lr, weight_decay=0.05, betas=(0.9, 0.999)),
    paramwise_cfg=dict(bias_decay_mult=0.0, norm_decay_mult=0.0),
    clip_grad=dict(max_norm=10.0))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1500),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.01,
        begin=2,
        end=max_epochs,
        T_max=max_epochs - 1,
        by_epoch=True),
]

# ----------------------------------------------------------------------------
# Runtime
# ----------------------------------------------------------------------------
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', interval=1, max_keep_ckpts=4, save_best=None,
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl', timeout=10800))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='mmdet.DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
randomness = dict(seed=0, deterministic=False)
# The ConvNeXt classification head is unused by the detector.
find_unused_parameters = True
# WeDetect-Tiny initialization (tools/fetch_checkpoints.sh), SHA256-verified.
load_from = 'trusted-wedetect://checkpoints/wedetect_tiny.pth'
resume = False
work_dir = './work_dirs/cytospm'
