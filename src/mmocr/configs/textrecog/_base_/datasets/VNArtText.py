##### UNCOMMENT THESE LINES TO ENABLE TRAIN BY LABELS.TXT FILE #####

img_prefix = "/PathToYourDatasetFolder/VNArtText"
root_dir = "/PathToYourDatasetFolder/VNArtText"

train_img_prefix = img_prefix
train_ann_file = f"{root_dir}/train_label.txt"
test_img_prefix = img_prefix
test_ann_file = f"{root_dir}/test_label.txt"

train = dict(
    type="RecogTextDataset",
    data_root=train_img_prefix,
    ann_file=train_ann_file,
    # loader=dict(
    #     type="AnnFileLoader",
    #     repeat=1,
    #     file_format="txt",
    #     parser=dict(
    #         type="LineStrParser",
    #         keys=["filename", "text"],
    #         keys_idx=[0, 1],
    #         separator="\t",
    #     ),
    # ),
    parser_cfg=dict(
        type='LineStrParser',
        keys=['filename', 'text'],
        keys_idx=[0, 1],
        separator="\t",
    ),
    pipeline=None,
    test_mode=False,
)

val = dict(
    type="RecogTextDataset",
    data_root=train_img_prefix,
    ann_file=test_ann_file,
    # loader=dict(
    #     type="AnnFileLoader",
    #     repeat=1,
    #     file_format="txt",
    #     parser=dict(
    #         type="LineStrParser",
    #         keys=["filename", "text"],
    #         keys_idx=[0, 1],
    #         separator="\t",
    #     ),
    # ),
    parser_cfg=dict(
        type='LineStrParser',
        keys=['filename', 'text'],
        keys_idx=[0, 1],
        separator="\t",
    ),
    pipeline=None,
    test_mode=False,
)

test = dict(
    type="RecogTextDataset",
    data_root=test_img_prefix,
    ann_file=test_ann_file,
    # loader=dict(
    #     type="AnnFileLoader",
    #     repeat=1,
    #     file_format="txt",
    #     parser=dict(
    #         type="LineStrParser",
    #         keys=["filename", "text"],
    #         keys_idx=[0, 1],
    #         separator="\t",
    #     ),
    # ),
    parser_cfg=dict(
        type='LineStrParser',
        keys=['filename', 'text'],
        keys_idx=[0, 1],
        separator="\t",
    ),
    pipeline=None,
    test_mode=True,
)

train_list = [train]
val_list = [val]
test_list = [test]
