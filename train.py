import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tools.cfg import py2cfg
import os
import torch
from torch import nn
import numpy as np
import argparse
from pathlib import Path
from tools.utils import Adder
from pytorch_lightning.loggers import CSVLogger
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch.nn.functional as F
from model.losses.clip_loss import ClipLoss


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)



def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", default=r'config/config.py')
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

class Supervision_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net
        self.automatic_optimization = False
        self.model_version = 'UDAIR'
        os.makedirs(self.config.weights_path, exist_ok=True)

        self.l1_loss = nn.L1Loss()
        self.contrastive_loss = ClipLoss()


        self.ssim_epoch_adder_train = Adder()
        self.psnr_epoch_adder_train = Adder()

        self.contrastive_loss_epoch_adder_train = Adder()
        self.contrastive_loss_epoch_adder_val = Adder()

        self.best_psnr = 0
        self.best_epoch = 0
        self.task_metrics = {}

        self.source_domain_features_sum_4 = None
        self.source_domain_features_sum_3 = None
        self.source_domain_features_sum_2 = None
        self.source_doamin_features_count = 0

        self.source_domain_distrib = None


    def forward(self, x):
        # only net is used in the prediction/inference
        results = self.net(x, tta=True)

        return results

    def training_step(self, batch, batch_idx):
        img, label, name, task = batch[0], batch[1], batch[2], batch[3]

        results = self.net(img)

        if type(results) is not torch.Tensor:
            pred = results[0]
            degrad_embeddings = results[1]
            logit_scale = results[2]
            degrad_distrib_4 = results[3]
            degrad_distrib_3 = results[4] 
            degrad_distrib_2 = results[5] 
        else:
            pred = results
            
        l1_loss = self.l1_loss(pred, label)

        num_tasks = len(self.config.source_domain_tasks)

        # Cross Sample Contrastive Learning
        batch_size = self.config.train_batch_size
        degradation_features = degrad_embeddings.view(batch_size, -1)
        cat_features = degradation_features.view(num_tasks, -1) 
        shuffled_degradation_features = degradation_features.view(num_tasks, batch_size // num_tasks, -1)
        shuffled_task_features = []
        for i in range(num_tasks):
            perm = torch.randperm(batch_size // num_tasks) 
            shuffled_task_features.append(shuffled_degradation_features[i, perm])
        shuffled_features = torch.cat(shuffled_task_features, dim=0) 
        shuffled_features = shuffled_features.view(num_tasks, -1)  
        contrastive_loss = self.contrastive_loss(cat_features, shuffled_features, logit_scale)

        pred_clip = torch.clamp(pred, 0, 1)
        pred = pred_clip.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        psnr = peak_signal_noise_ratio(pred, label)
        self.psnr_epoch_adder_train(psnr.item())
        
        loss = l1_loss + contrastive_loss*0.2

        # save degradation feature distribution
        task_distrib_batch_4 = degrad_distrib_4.view(num_tasks, degrad_distrib_4.shape[0] // num_tasks, *degrad_distrib_4.shape[1:]).detach() 
        task_distrib_batch_3 = degrad_distrib_3.view(num_tasks, degrad_distrib_3.shape[0] // num_tasks, *degrad_distrib_3.shape[1:]).detach()
        task_distrib_batch_2 = degrad_distrib_2.view(num_tasks, degrad_distrib_2.shape[0] // num_tasks, *degrad_distrib_2.shape[1:]).detach()

        task_distrib_batch_mean_4 = torch.mean(task_distrib_batch_4, dim=1).squeeze() 
        task_distrib_batch_mean_3 = torch.mean(task_distrib_batch_3, dim=1).squeeze()
        task_distrib_batch_mean_2 = torch.mean(task_distrib_batch_2, dim=1).squeeze()

        if self.source_domain_features_sum_4 is None:
            self.source_domain_features_sum_4 = task_distrib_batch_mean_4
            self.source_domain_features_sum_3 = task_distrib_batch_mean_3
            self.source_domain_features_sum_2 = task_distrib_batch_mean_2
        else:
            self.source_domain_features_sum_4 += task_distrib_batch_mean_4
            self.source_domain_features_sum_3 += task_distrib_batch_mean_3
            self.source_domain_features_sum_2 += task_distrib_batch_mean_2
        self.source_doamin_features_count += 1

        # supervision stage
        opt = self.optimizers(use_pl_optimizer=False)
        self.manual_backward(loss)
        if (batch_idx + 1) % self.config.accumulate_n == 0:
            opt.step()
            opt.zero_grad() 

        sch = self.lr_schedulers()
        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % 1 == 0:
            sch.step()

        return {'loss': loss, 'l1_loss': l1_loss.detach(), 'contrastive_loss': contrastive_loss.detach()}
        

    def training_epoch_end(self, outputs):
        psnr = self.psnr_epoch_adder_train.average()
        
        print('\n')

        loss = torch.stack([x['loss'] for x in outputs]).mean()
        l1_loss = torch.stack([x['l1_loss'] for x in outputs]).mean()
        contrastive_loss = torch.stack([x['contrastive_loss'] for x in outputs]).mean()


        print(f"Epoch {self.current_epoch}, Loss: {loss:.3f}, L1 Loss: {l1_loss:.3f}, Contrastive Loss: {contrastive_loss:.3f}")

        log_dict = {'train_loss': loss, 'train_psnr': psnr}
        self.log_dict(log_dict, prog_bar=True)

        self.ssim_epoch_adder_train.reset()
        self.psnr_epoch_adder_train.reset()


    def validation_step(self, batch, batch_idx):
        img, label, name, task = batch[0], batch[1], batch[2], batch[3]
        if self.config.val_image_size is None:
            img, padding = pad_img_tensor(img)

        with torch.no_grad():
            results = self.forward(img)

        if type(results) is not torch.Tensor:
            pred = results[0]
            degrad_embeddings = results[1]
        else:
            pred = results

        if self.config.val_image_size is None:
            pred = unpad_img_tensor(pred, padding)


        l1_loss = self.l1_loss(pred, label)


        val_loss = l1_loss 


        pred = pred.detach().cpu().numpy()
        label = label.detach().cpu().numpy()

        task_list = list(task)
        unique_tasks = list(set(task_list))

        batch_metrics = {}

        for current_task in unique_tasks:
            indices = [i for i, t in enumerate(task_list) if t == current_task]

            pred_task = np.clip(pred[indices], 0, 1)
            label_task = label[indices]

            ssim_list = []
            for i in range(pred_task.shape[0]):
                ssim_list.append(structural_similarity(pred_task[i].squeeze(), label_task[i].squeeze(), channel_axis=0, data_range=1))
            ssim_value = np.mean(ssim_list)
            psnr_value = peak_signal_noise_ratio(pred_task, label_task)

            if current_task not in batch_metrics:
                batch_metrics[current_task] = {'ssim': [], 'psnr': []}
            batch_metrics[current_task]['ssim'].append(ssim_value.item())
            batch_metrics[current_task]['psnr'].append(psnr_value)

        for task_name, metrics in batch_metrics.items():
            if task_name not in self.task_metrics:
                self.task_metrics[task_name] = {'ssim': [], 'psnr': []}
            self.task_metrics[task_name]['ssim'].extend(metrics['ssim'])
            self.task_metrics[task_name]['psnr'].extend(metrics['psnr'])


        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        avg_ssim = 0
        avg_psnr = 0
        for task_name, metrics in self.task_metrics.items():
            task_avg_ssim = np.mean(metrics['ssim'])
            task_avg_psnr = np.mean(metrics['psnr'])
            print(f"\nTask: {task_name}, Average SSIM: {task_avg_ssim:.3f}, Average PSNR: {task_avg_psnr:.3f}")
            avg_ssim += task_avg_ssim
            avg_psnr += task_avg_psnr

        avg_ssim /= len(self.task_metrics)
        avg_psnr /= len(self.task_metrics)

        loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        log_dict = {'val_loss': loss, 'val_ssim': avg_ssim, 'val_psnr': avg_psnr}
        self.log_dict(log_dict, prog_bar=True)
        self.task_metrics.clear()

        # save best distribution
        if self.source_domain_features_sum_4 is not None:
            self.source_domain_features_sum_4 = self.source_domain_features_sum_4 / self.source_doamin_features_count
            self.source_domain_features_sum_3 = self.source_domain_features_sum_3 / self.source_doamin_features_count
            self.source_domain_features_sum_2 = self.source_domain_features_sum_2 / self.source_doamin_features_count
            
            self.source_domain_distrib = {
                'distrib_4': self.source_domain_features_sum_4,
                'distrib_3': self.source_domain_features_sum_3,
                'distrib_2': self.source_domain_features_sum_2,
            }
        
        self.source_domain_features_sum_4 = None
        self.source_domain_features_sum_3 = None
        self.source_domain_features_sum_2 = None
        self.source_doamin_features_count = 0


        if avg_psnr > self.best_psnr and not self.trainer.sanity_checking:
            self.best_psnr = avg_psnr
            self.best_epoch = self.current_epoch
            if self.source_domain_distrib is not None:
                torch.save(self.source_domain_distrib, os.path.join(self.config.weights_path, 'source_domain_distrib.pth'))
                print('\nThe best source domain distribution is saved.')


        print(f"\nBest Average PSNR: {self.best_psnr:.3f} at epoch {self.best_epoch}")


    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler

        return [optimizer], [lr_scheduler]

    def train_dataloader(self):

        return self.config.train_loader

    def val_dataloader(self):

        return self.config.val_loader
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint['best_psnr'] = self.best_psnr
        checkpoint['best_epoch'] = self.best_epoch

    def on_load_checkpoint(self, checkpoint):
        self.best_psnr = checkpoint.get('best_psnr', 0)
        self.best_epoch = checkpoint.get('best_epoch', 0)
    


# training
def main():
    args = get_args()
    config = py2cfg(args.config_path)
    if config.train_loader is None:
        raise FileNotFoundError(f"Training data not found: {config.train_dataset_path}")
    if config.val_loader is None:
        raise FileNotFoundError(f"Validation data not found: {config.val_dataset_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but this training script is configured with accelerator='gpu'.")
    rand_seed = 123
    seed_everything(rand_seed)
    print('seed: ', str(rand_seed))


    checkpoint_callback = ModelCheckpoint(save_top_k=config.save_top_k, monitor=config.monitor,
                                          save_last=config.save_last, mode=config.monitor_mode,
                                          dirpath=config.weights_path,
                                          filename=config.weights_name)
    logger = CSVLogger('lightning_logs', name=config.log_name)

    model = Supervision_Train(config)
    if config.pretrained_ckpt_path:
        model = Supervision_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)

    trainer = pl.Trainer(devices=config.gpus, max_epochs=config.max_epoch, accelerator='gpu',
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         callbacks=checkpoint_callback, strategy=config.strategy,
                         resume_from_checkpoint=config.resume_ckpt_path,
                         logger=logger)
    trainer.fit(model=model)
    # trainer.fit(model=model, ckpt_path=config.resume_ckpt_path)



if __name__ == "__main__":
   main()


