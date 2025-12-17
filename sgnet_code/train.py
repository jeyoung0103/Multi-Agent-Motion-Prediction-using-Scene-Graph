from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger

from datamodules import ETRIDataModule
from predictors import QCNet, QCNetWithGraphormer

if __name__ == '__main__':
    pl.seed_everything(2023, workers=True)
    
    # Tensor Core optimization for NVIDIA RTX A5000
    import torch
    torch.set_float32_matmul_precision('medium')

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default="/workspace/datasets")
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8, help='Increased for better data loading performance')
    parser.add_argument('--devices', type=int, default=2, help='The number of possible GPU devices')
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--train_processed_dir', type=str, default="train_sgnet")
    parser.add_argument('--val_processed_dir', type=str, default="val_sgnet")
    parser.add_argument('--test_processed_dir', type=str, default="test_sgnet")
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--max_epochs', type=int, default=64, help='Increased for better convergence with Graphormer')      
    parser.add_argument('--dataset', type=str, default='ETRI_Dataset', help='DO NOT ALTER THIS')
    parser.add_argument('--num_historical_steps', type=int, default=20, help='DO NOT ALTER THIS')
    parser.add_argument('--num_future_steps', type=int, default=60, help='DO NOT ALTER THIS')
    parser.add_argument('--num_recurrent_steps', type=int, default=3, help='DO NOT ALTER THIS')
    parser.add_argument('--pl2pl_radius', type=int, default=150)
    parser.add_argument('--pl2a_radius', type=int, default=50)
    parser.add_argument('--a2a_radius', type=int, default=50)
    parser.add_argument('--pl2m_radius', type=int, default=150)
    parser.add_argument('--a2m_radius', type=int, default=150)
    parser.add_argument('--num_t2m_steps', type=int, default=10)
    parser.add_argument('--time_span', type=int, default=10)
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--output_dim', type=int, default=2)
    parser.add_argument('--output_head', action='store_true')
    parser.add_argument('--num_modes', type=int, default=6, help='DO NOT ALTER THIS')
    parser.add_argument('--num_freq_bands', type=int, default=32, help='Reduced for better efficiency')
    parser.add_argument('--num_map_layers', type=int, default=2, help='Increased for better map encoding')
    parser.add_argument('--num_agent_layers', type=int, default=3, help='Increased for better agent encoding')
    parser.add_argument('--num_dec_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--head_dim', type=int, default=20, help='Adjusted for hidden_dim=160')
    parser.add_argument('--dropout', type=float, default=0.15, help='Increased for better regularization')
    parser.add_argument('--lr', type=float, default=1.5e-4, help='Reduced for more stable training')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Increased for better regularization')
    parser.add_argument('--T_max', type=int, default=80, help='Adjusted for max_epochs=80')
    
    # Graphormer parameters
    parser.add_argument('--use_graphormer', action='store_true', help='Use Graphormer for agent-agent relationships')
    parser.add_argument('--num_agent_types', type=int, default=3, help='Number of agent types for Graphormer (ETRI: vehicle, pedestrian, cyclist)')
    parser.add_argument('--num_speed_bins', type=int, default=40, help='Increased speed bins for finer granularity (0-40 km/h range)')
    parser.add_argument('--num_spatial_bins', type=int, default=80, help='Increased spatial bins for better spatial resolution (400m range)')
    parser.add_argument('--num_ttc_bins', type=int, default=30, help='Increased TTC bins for finer temporal resolution (0-6s range)')
    parser.add_argument('--num_edge_types', type=int, default=5, help='Number of edge types for Graphormer')
    parser.add_argument('--graphormer_history_steps', type=int, default=2, help='Number of recent steps for dual-stage Graphormer')
    parser.add_argument('--graphormer_history_pos_eps', type=float, default=0.5, help='Minimum positional change (meters) to keep nodes in older Graphormer passes')
    
    # # Additional training parameters
    # parser.add_argument('--warmup_epochs', type=int, default=5, help='Learning rate warmup epochs')
    # parser.add_argument('--gradient_clip_val', type=float, default=1.0, help='Gradient clipping value')
    # parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Gradient accumulation batches')
    # parser.add_argument('--precision', type=str, default='16-mixed', help='Mixed precision training for efficiency')
    # parser.add_argument('--sync_dist', type=bool, default=True, help='Sync distributed metrics')
    
    # WandB 설정
    parser.add_argument('--wandb_project', type=str, default='ETRI', help='WandB project name')
    parser.add_argument('--wandb_name', type=str, default='SGNet2', help='WandB run name')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=['QCNet', 'trajectory_prediction'], help='WandB tags')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    
    args = parser.parse_args()

    datamodule = {'ETRI_Dataset': ETRIDataModule, }[args.dataset](**vars(args))
    
    # Choose model based on Graphormer usage
    if args.use_graphormer:
        model = QCNetWithGraphormer(**vars(args))
        print("🚀 Using QCNet with Graphormer integration!")
    else:
        model = QCNet(**vars(args))
        print("📊 Using standard QCNet model")
    model_checkpoint = ModelCheckpoint(monitor='val_minADE', save_top_k=5, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # WandB Logger 설정
    logger = None
    if args.use_wandb:
        # 실행 이름 자동 생성 (지정되지 않은 경우)
        if args.wandb_name is None:
            args.wandb_name = f"QCNet_ep{args.max_epochs}_bs{args.train_batch_size}_lr{args.lr}"
        
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_name,
            tags=args.wandb_tags,
            log_model='all',  # 모델 체크포인트도 wandb에 저장
            save_dir='wandb_logs'
        )
        
        # 하이퍼파라미터 로깅
        hyperparams = {
            'model': 'QCNetWithGraphormer' if args.use_graphormer else 'QCNet',
            'dataset': 'ETRI_Dataset',
            'batch_size': args.train_batch_size,
            'learning_rate': args.lr,
            'max_epochs': args.max_epochs,
            'num_modes': args.num_modes,
            'hidden_dim': args.hidden_dim,
            'devices': args.devices,
            'use_graphormer': args.use_graphormer
        }
        
        # Add Graphormer-specific parameters if enabled
        if args.use_graphormer:
            hyperparams.update({
                'num_agent_types': args.num_agent_types,
                'num_speed_bins': args.num_speed_bins,
                'num_spatial_bins': args.num_spatial_bins,
                'num_ttc_bins': args.num_ttc_bins,
                'num_edge_types': args.num_edge_types,
                # 'warmup_epochs': args.warmup_epochs,
                # 'gradient_clip_val': args.gradient_clip_val,
                # 'precision': args.precision
            })
        
        logger.log_hyperparams(hyperparams)
        
        print(f"🚀 WandB logging enabled! Project: {args.wandb_project}, Run: {args.wandb_name}")
    
    # DDP strategy with unused parameters detection for Graphormer integration
    if args.use_graphormer:
        strategy = DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True)
        print("🔧 Using DDP with unused parameters detection for Graphormer integration")
    else:
        strategy = DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)
        print("📊 Using standard DDP strategy")
    
    trainer = pl.Trainer(
        accelerator=args.accelerator, 
        devices=args.devices,
        strategy=strategy,
        callbacks=[model_checkpoint, lr_monitor], 
        max_epochs=args.max_epochs,
        logger=logger,
        # precision=args.precision,
        # gradient_clip_val=args.gradient_clip_val,
        # accumulate_grad_batches=args.accumulate_grad_batches,
        # sync_dist=args.sync_dist
    )
    trainer.fit(model, datamodule)
