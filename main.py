import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from AnimeDiffusion import AnimeDiffusion
from distutils.util import strtobool

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training Configuration Parser')
    
    # Training Configuration
    parser.add_argument('--do_train', type=lambda x: bool(strtobool(x)), default=True, 
                    help='Enable or disable training (True/False)')
    parser.add_argument('--do_finetuning', type=lambda x: bool(strtobool(x)), default=False, 
                    help='Enable or disable finetuning (True/False)')
    parser.add_argument('--do_test', type=lambda x: bool(strtobool(x)), default=False, 
                        help='Enable or disable test (True/False)')
    parser.add_argument('--epochs', type=int, default=5, 
                        help='Number of training epochs')
    parser.add_argument('--test_output_dir', type=str, 
                        default='./result/', 
                        help='Directory for test outputs')
    
    # Diffusion Process Configuration
    parser.add_argument('--time_step', type=int, default=1000, 
                        help='Number of diffusion time steps')
    
    # Betas Configuration
    parser.add_argument('--linear_start', type=float, default=1e-6, 
                        help='Starting value for linear beta schedule')
    parser.add_argument('--linear_end', type=float, default=1e-2, 
                        help='Ending value for linear beta schedule')
    
    # UNet Configuration
    parser.add_argument('--channel_in', type=int, default=7, 
                        help='Input channels for UNet')
    parser.add_argument('--channel_out', type=int, default=3, 
                        help='Output channels for UNet')
    parser.add_argument('--channel_mult', nargs='+', type=int, 
                        default=[1, 2, 4, 8], 
                        help='Channel multipliers for UNet')
    parser.add_argument('--attention_head', type=int, default=4, 
                        help='Number of attention heads')
    parser.add_argument('--cbam', type=bool, default=False, 
                        help='Enable or disable CBAM')
    
    # Optimizer Configuration
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-8, 
                        help='Minimum learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=1, 
                        help='Number of warmup epochs')
    parser.add_argument('--weight_decay', type=float, default=0.01, 
                        help='Weight decay value')
    
    # Data Paths
    parser.add_argument('--train_reference_path', type=str, 
                        default='/data/Anime/train_data/reference/', 
                        help='Path to reference data')
    parser.add_argument('--train_condition_path', type=str, 
                        default='/data/Anime/train_data/sketch/', 
                        help='Path to condition data')
    parser.add_argument('--test_reference_path', type=str, 
                        default='/data/Anime/test_data_shuffled/reference/', 
                        help='Path to reference data')
    parser.add_argument('--test_condition_path', type=str, 
                        default='/data/Anime/test_data_shuffled/sketch/', 
                        help='Path to condition data')
    
    # Batch Sizes
    parser.add_argument('--train_batch_size', type=int, default=32, 
                        help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=1, 
                        help='Batch size for validation')
    
    # Image Size
    parser.add_argument('--size', type=int, default=256, 
                        help='Image size')
    
    parser.add_argument('--gpus', nargs='+', type=int, default=[1])
    
    # Parse arguments
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    # Parse arguments
    cfg = parse_arguments()

    # Set seed for reproducibility
    pl.seed_everything(42)

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="train_avg_loss",
        filename='{epoch:02d}-{train_avg_loss:.4f}',
        save_top_k=3,
        mode="min",
        every_n_epochs=1,
    )

    # TensorBoard logger
    tb_logger = TensorBoardLogger("logs")

    # Trainer configuration
    trainer = pl.Trainer(
        default_root_dir="./",
        devices=cfg.gpus,
        accelerator="cuda",
        precision="16-mixed",
        max_epochs=cfg.epochs,  # Use cfg.epochs instead of cfg.max_epochs
        reload_dataloaders_every_n_epochs=0,
        num_sanity_val_steps=0,
        logger=[tb_logger],
        callbacks=[checkpoint_callback],
        strategy="ddp_find_unused_parameters_true",
    )

    model = AnimeDiffusion(cfg)

    # Training
    if cfg.do_train:
        if cfg.do_finetuning:
            trainer.fit(model, ckpt_path="/root/AnimeDiffusion/logs/lightning_logs/version_0/checkpoints/epoch=04-train_loss=0.0139.ckpt")
        else: 
            trainer.fit(model)
    

    # Testing
    if cfg.do_test:
        os.makedirs(cfg.test_output_dir, exist_ok=True)  # 디렉토리 생성
        trainer.test(model, ckpt_path="/root/AnimeDiffusion/logs/lightning_logs/version_0/checkpoints/epoch=04-train_loss=0.0139.ckpt")