from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp
config = edict()
config.margin_list = (1.0, 0.5, 0.0)  # arcface
config.network = "vit_l_dp005_mask_005"
config.resume = True
config.checkpoint = "../../../checkpoints/idc/transface_ms1mv2_L.pt"
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.weight_decay = 0.1
config.batch_size = 128
config.optimizer = "adamw"
config.lr = 0.001
config.verbose = 2000
config.dali = False

config.save_all_states = True

config.rec = "/4tb/datasets/multi-pie_crop_v2/train"
config.num_classes = 200
config.num_image = 59862
config.num_epoch = 30
config.warmup_epoch = config.num_epoch // 10
config.val_targets = []
