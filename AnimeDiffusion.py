import os
import time
import torch
import math
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data.anime import Anime
from models.diffusion import GaussianDiffusion
import utils.image
import utils.path
import torch.nn.functional as F

class AnimeDiffusion(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.automatic_optimization = False
        self.save_hyperparameters(cfg)
        
        betas = {
            "linear_start": self.cfg.linear_start,
            "linear_end": self.cfg.linear_end,
        }
        
        unet = {
            "channel_in": self.cfg.channel_in,
            "channel_out": self.cfg.channel_out,
            "channel_mult": self.cfg.channel_mult,
            "attention_head": self.cfg.attention_head,
            "cbam": self.cfg.cbam,
        }
        
        self.model = GaussianDiffusion(
            time_step=cfg.time_step,
            betas=betas,
            unet=unet
        )
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr * len(self.cfg.gpus),
            weight_decay=self.cfg.weight_decay
        )

        def lr_lambda(step):
            epoch = (step / len(self.train_dataloader()) + self.current_epoch)
            
            # Warmup
            if epoch < self.cfg.warmup_epochs:
                warmup_ratio = epoch / self.cfg.warmup_epochs
                return warmup_ratio * (1 - self.cfg.min_lr/self.cfg.lr) + self.cfg.min_lr/self.cfg.lr

            # Cosine Decay
            progress = (epoch - self.cfg.warmup_epochs) / (self.cfg.epochs - self.cfg.warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress)) * (1 - self.cfg.min_lr/self.cfg.lr) + self.cfg.min_lr/self.cfg.lr

        # Scheduler
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lr_lambda
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def train_dataloader(self):
        self.train_dataset = Anime(
            reference_path = self.cfg.train_reference_path, 
            condition_path = self.cfg.train_condition_path,
            size = self.cfg.size,
        )
        train_dataloader = DataLoader(
            self.train_dataset, 
            batch_size = self.cfg.train_batch_size, 
            shuffle = True, 
            pin_memory=True,
            drop_last=True
        )
        return train_dataloader

    def test_dataloader(self):
        self.test_dataset = Anime(
            reference_path = self.cfg.test_reference_path, 
            condition_path = self.cfg.test_condition_path,
            size = self.cfg.size,
        )
        test_dataset = DataLoader(
            self.test_dataset, 
            batch_size = self.cfg.test_batch_size, 
            shuffle = False, 
            pin_memory=True,
            drop_last=True
        )
        return test_dataset
    
    def on_train_start(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.print(f"Total Parameters: {total_params:,}")
        self.print(f"Trainable Parameters: {trainable_params:,}")

    
    def pretraining_step(self, batch, batch_idx):
        x_ref = batch["reference"].to(self.device)  # [B, 3, H, W]
        x_con = batch["condition"].to(self.device)  # [B, 1, H, W]
        x_dis = batch["distorted"].to(self.device)  # [B, 3, H, W]

        batch_size = x_ref.size(0)
        t = torch.randint(0, self.cfg.time_step, (batch_size,), device=self.device).long()  # [B]

        # [B, 1, H, W] + [B, 3, H, W] → [B, 4, H, W]
        x_cond = torch.cat([x_con, x_dis], dim=1)

        loss = self.model(x_ref, t, x_cond)

        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        
        scheduler = self.lr_schedulers()
        scheduler.step()

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        current_lr = optimizer.param_groups[0]['lr']
        self.log('lr', current_lr, prog_bar=True, logger=True)
        
        return loss
    
    def finetuning_step(self, batch, batch_idx):
        x_ref = batch["reference"].to(self.device)  # [B, 3, H, W]
        x_con = batch["condition"].to(self.device)  # [B, 1, H, W]
        x_dis = batch["distorted"].to(self.device)  # [B, 3, H, W]

        batch_size = x_ref.size(0)
        t = torch.randint(0, self.cfg.time_step, (batch_size,), device=self.device).long()  # [B]

        # [B, 1, H, W] + [B, 3, H, W] → [B, 4, H, W]
        x_cond = torch.cat([x_con, x_dis], dim=1)

        with torch.no_grad():
            x_T = self.model.fix_forward(x_ref, x_cond=x_cond)

        x_til = self.model.inference_ddim(x_t=x_T, x_cond=x_cond)[-1]
        loss = F.mse_loss(x_til, x_ref)

        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        
        scheduler = self.lr_schedulers()
        scheduler.step()

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        current_lr = optimizer.param_groups[0]['lr']
        self.log('lr', current_lr, prog_bar=True, logger=True)
        
        return loss

    
    def training_step(self, batch, batch_idx):
        if self.cfg.do_finetuning:
            return self.finetuning_step(batch, batch_idx)
        else:
            return self.pretraining_step(batch, batch_idx)

    def on_train_epoch_end(self):
        avg_loss = self.all_gather(self.trainer.callback_metrics["train_loss"]).mean()
        
        if self.trainer.is_global_zero:
            self.print(f"Epoch {self.current_epoch} - Avg Loss: {avg_loss:.4f}")
        
        self.log("train_avg_loss", avg_loss, prog_bar=True)
            
    def test_step(self, batch, batch_idx):
        x_ref = batch["reference"].to(self.device)  # [B, 3, H, W]
        x_con = batch["condition"].to(self.device)  # [B, 1, H, W]
        x_dis = batch["distorted"].to(self.device)  # [B, 3, H, W]

        noise = torch.randn_like(x_ref).to(self.device)  # [B, 3, H, W]

        with torch.no_grad():
            rets = self.model.inference_ddim(
                x_t=noise,
                x_cond=torch.cat([x_con, x_dis], dim=1),
                time_steps=50
            )[-1]

        images = utils.image.tensor2PIL(rets)
        for i, filename in enumerate(batch['name']):
            output_path = os.path.join(self.cfg.test_output_dir, f'ret_{filename}')
            images[i].save(output_path)

    def on_test_epoch_end(self):
        self.print(f"All test outputs saved to {self.cfg.test_output_dir}")
