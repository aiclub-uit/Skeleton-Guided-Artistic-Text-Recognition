_base_ = [
    '../_base_/datasets/VNArtText.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_step_5e.py',
    '_base_satrn_shallow_vnarttext.py',
]

load_from = "weights/satrn_shallow_5e_st_mj_20220915_152443-5fd04a4c.pth"

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=5)
checkpoint_config = dict(interval=5)

param_scheduler = [
    dict(type='MultiStepLR', milestones=[15, 18], end=20),
]

# dataset settings
train_list = _base_.train_list
test_list = _base_.test_list
train_dataset = dict(
    type='ConcatDataset', datasets=train_list, pipeline=_base_.train_pipeline)
test_dataset = dict(
    type='ConcatDataset', datasets=test_list, pipeline=_base_.test_pipeline)

# optimizer
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='Adam', lr=3e-4))

batch_size = 42

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

test_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)

val_dataloader = test_dataloader

val_evaluator = dict(
    dataset_prefixes=[
        'VNArtText',
    ],
)
test_evaluator = val_evaluator

auto_scale_lr = dict(base_batch_size=batch_size * 1)
