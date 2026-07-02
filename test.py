import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import os
import random
from pathlib import Path
from tqdm import tqdm
from model.tent import tent
from train import Supervision_Train
from tools.cfg import py2cfg

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.deterministic = True

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, default=r'config/config.py', help="Path to config")
    arg("-o", "--output_path", type=Path, default=r'results', help="Path where to save resulting masks.")
    return parser.parse_args()


def pad_img_tensor(img: torch.Tensor, base: int = 16):
    squeeze_after = False
    if img.dim() == 3:
        img = img.unsqueeze(0)
        squeeze_after = True

    _, C, H, W = img.shape

    pad_h = (base - H % base) if H % base != 0 else 0
    pad_w = (base - W % base) if W % base != 0 else 0

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded_img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
    padding = (pad_left, pad_right, pad_top, pad_bottom)

    if squeeze_after:
        padded_img = padded_img.squeeze(0)
    return padded_img, padding

def unpad_img_tensor(img: torch.Tensor, padding):
    pad_left, pad_right, pad_top, pad_bottom = padding

    if img.dim() == 4:
        return img[:, :, pad_top: img.shape[2]-pad_bottom, pad_left: img.shape[3]-pad_right]
    else:
        return img[:, pad_top: img.shape[1]-pad_bottom, pad_left: img.shape[2]-pad_right]
    

def main():
    seed_everything(42)
    args = get_args()
    config = py2cfg(args.config_path)
    output_path = os.path.join(args.output_path, config.test_output_path)
    os.makedirs(output_path, exist_ok=True)
    model = Supervision_Train.load_from_checkpoint(os.path.join(config.weights_path, config.test_weights_name+'.ckpt'), config=config)
    model.cuda(config.gpus[0])
    model.eval()


    test_loader = config.test_loader

    inf_time = 0
    count = 0


    if config.tta:
        adaptation_profile = config.adaptation_profile
        model = tent.configure_model(model, modules_identifier="distrib_adapt")
        params, param_names = tent.collect_params(model, modules_identifier="distrib_adapt")
        print("params for adaptation:", param_names)

        optimizer = torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, fused=True)
        tent_model = tent.Tent(model, optimizer, steps=adaptation_profile['steps'], episodic=False)
        
        profile_anchor = adaptation_profile['reference_index']

        source_domain_distrib = torch.load(os.path.join(config.weights_path, adaptation_profile['feature_bank']))
        source_domain_distrib_4 = source_domain_distrib['distrib_4'].cuda(config.gpus[0])
        source_domain_distrib_3 = source_domain_distrib['distrib_3'].cuda(config.gpus[0])
        source_domain_distrib_2 = source_domain_distrib['distrib_2'].cuda(config.gpus[0])
        source_domain_distrib_list = [source_domain_distrib_4, source_domain_distrib_3, source_domain_distrib_2]
    

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # warm up
    real_batch, _ = next(iter(test_loader))
    real_batch = real_batch.cuda(config.gpus[0])
    print("Warm up with REAL data...")
    for i in range(10):
        if config.tta:
            tent_model.reset()
            with torch.amp.autocast(device_type='cuda', dtype=dtype):
                results = tent_model(real_batch, source_domain_distrib_list, profile_anchor)
        else:
            with torch.no_grad():
                results = model(real_batch)[0]

    for i, (img, filename) in enumerate(tqdm(test_loader)):
        if config.test_image_size is None:
            img, padding = pad_img_tensor(img)

        img = img.cuda(config.gpus[0], non_blocking=True)

        tic = time.time()
        if config.tta:
            tent_model.reset()
            with torch.amp.autocast(device_type='cuda', dtype=dtype):
                results = tent_model(img, source_domain_distrib_list, profile_anchor)
        else:
            with torch.no_grad():
                results = model(img)[0]
        toc = time.time()

        if type(results) is not torch.Tensor:
            pred = results[0]
        else:
            pred = results
        pred_time = toc - tic
        inf_time += pred_time
        count += 1

        if config.test_image_size is None:
            pred = unpad_img_tensor(pred, padding)

        if pred.shape[0] == 1:
            torchvision.utils.save_image(pred, os.path.join(output_path, filename[0]))
        else:
            for b in range(pred.shape[0]):
                torchvision.utils.save_image(pred[b], os.path.join(output_path, filename[b]))
    
    print('Avg inference time: {} s'.format(inf_time/count))


if __name__ == "__main__":
    main()
