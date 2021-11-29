from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.gm = True
config.loss = "lgmface"
config.network = "r50"
config.alpha = 0.01
config.lambdaa = 0.00003
config.resume = False
config.output = 'work_dirs/vggface_l0.00003_a0.01'
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.1  # batch size is 512

config.rec = "train_tmp/faces_vgg_112"
config.num_classes = 9114
config.num_image = 3228974
config.num_epoch = 12
config.warmup_epoch = -1
config.decay_epoch = [10, 16, 22]
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
