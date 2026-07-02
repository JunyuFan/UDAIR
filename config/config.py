import os
import torch
from model.models.UDAIR import UDAIR
from model.datasets.data import *
from model.tent.tent_profile import _make_tta_profile

source_domain_tasks = ['denoising', 'dehazing', 'deraining', 'low-light', 'underwater']
target_domain_tasks = ['denoising_polyu', 'dehazing_urhi', 'deraining_lhp', 'low-light_lime', 'underwater_ufo']

max_epoch = 2000
train_batch_size = len(source_domain_tasks) * 2  # Need to be a multiple of the number of tasks
val_batch_size = len(source_domain_tasks)
test_batch_size = 1
lr = 1e-4
accumulate_n = 1

weights_name = "UDAIR"
weights_path = "model_weights/UDAIR"
test_weights_name = "UDAIR"
log_name = 'IR/{}'.format(weights_name)
monitor = 'val_psnr'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
gpus = [0]
strategy = None
pretrained_ckpt_path = None
resume_ckpt_path = None

test_dataset = 'denoising_polyu'

output_dir_version = ''
test_output_path = test_dataset + output_dir_version

val_image_size = 256
test_image_size = 512

tta_steps = 5
tta_image_size = 256
tta_feature_bank = 'source_domain_distrib.pth'

adaptation_profile = _make_tta_profile(
    test_dataset,
    source_domain_tasks,
    steps=tta_steps,
    image_size=tta_image_size,
    feature_bank=tta_feature_bank,
)
tta = adaptation_profile['enabled']
if tta:
    test_image_size = adaptation_profile['image_size']

#  define the network
net = UDAIR()



# define the dataloader
train_dataset_path = 'data/train'
val_dataset_path = 'data/val'
test_dataset_path = 'data/test'

train_loader = None
if os.path.isdir(train_dataset_path):
    train_loader = get_train_loader(dataset_path=train_dataset_path, tasks=source_domain_tasks, patch_size=128, data_augmentation=True, batch_size=train_batch_size, num_workers=32, pin_memory=True, shuffle=False, drop_last=True)

val_loader = None
if os.path.isdir(val_dataset_path):
    val_loader = get_eval_loader(dataset_path=val_dataset_path, tasks=source_domain_tasks, image_size=val_image_size, batch_size=val_batch_size, num_workers=32, shuffle=False)

test_loader = None
if os.path.isdir(os.path.join(test_dataset_path, test_dataset, 'images')):
    test_loader = get_test_loader(dataset_path=test_dataset_path, test_dataset=test_dataset, image_size=test_image_size, batch_size=test_batch_size, num_workers=32) # image_size: None for adaptive crop, or give a size for resize



# define the optimizer
optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
