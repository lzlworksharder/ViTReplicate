from dataset import load_train_val_cifar100
from train import TrainLoop
from torch.optim import SGD
from torch.optim import lr_scheduler
from transformers import AutoModelForImageClassification
from torchvision import transforms
from torch import nn
from argparse import ArgumentParser
import optuna
def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def main():
    seed_everything(42)


    image_size = 384
    train_trans = transforms.Compose([
            transforms.Resize((448,448)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
        ])

    test_trans=transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
        ])


    parser = ArgumentParser()
    parser.add_argument(
        '--debug',
        action='store_true',
    )
    parser.add_argument(
        '--freeze_feature_extractor',
        action='store_true',
        help='Freeze the feature extractor of the model',
    )

    args = parser.parse_args()
    debug = args.debug

    config = {
        "steps": 10000 if not debug else 10,
        "eval_step": 50 if not debug else 10,
        "log_step": 5,
        # train_micro_batch_per_device=32
        # total_batch_size = micro_per_device*accumu_steps*num_gpus
        'batch_size': 32 if not debug else 4,
        'gradient_accumulation_steps': 1,
        'num_workers': 4 if not debug else 0,
        'data_dir': "./data/cifar100",
        'model_dir':'./pretrained/vit-l-16',
        'freeze_feature_extractor': args.freeze_feature_extractor,
        "precision": "fp16",
        'early_stopping': False,
        }
    num_gpus = 8
    config['steps'] = config['batch_size']*config['gradient_accumulation_steps']*num_gpus if not debug else 10

    print("Loading data...")
    train_loader, val_loader = load_train_val_cifar100(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        train_trans=train_trans,
        test_trans=test_trans,
    )

    classes = train_loader.dataset.features['fine_label'].names
    cls2idx = {cls: idx for idx, cls in enumerate(classes)}
    idx2cls = {idx: cls for idx, cls in enumerate(classes)}



    def objective(trial):
        # Define the hyperparameters to tune
        lr = trial.suggest_categorical("lr", [1e-3, 3e-3, 1e-2,3e-2])

        # Create a fresh model instance for each trial
        model = AutoModelForImageClassification.from_pretrained(
            config['model_dir'],
            num_labels=len(classes),
            ignore_mismatched_sizes=True,
            image_size=image_size,
            # attn_implementation='eager',
        )
        # model.gradient_checkpointing_enable()

        if config['freeze_feature_extractor']:
            for name, param in model.named_parameters():
                if not 'classifier' in name:
                    param.requires_grad = False

        # Create the optimizer with the suggested hyperparameters
        optimizer = SGD(model.parameters(), lr=lr, weight_decay=0, momentum=0.9)

        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['steps'],
            eta_min=0,
        )

        train_args = {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "steps": config["steps"],
            "eval_step":config["eval_step"],
            "log_step": config["log_step"],
            "precision": config["precision"],
            'debug':debug,
            'early_stopping':config['early_stopping'],
            'num_classes':len(classes),
            'gradient_accumulation_steps': config['gradient_accumulation_steps'],
            # 'trial': trial,
            # 'patience':4,
            # 'min_delta':0.001,
        }
        loop = TrainLoop(**train_args)
        print("Training loop object initialized")

        print("Starting training...")
        loop.run_loop()
        
        return loop.acc.compute().item()
    search_space = {
    "lr": [1e-3, 3e-3, 1e-2, 3e-2],
    # "batch_size": [16, 32, 64],
    # "momentum": [0.9, 0.95, 0.99],
    # "weight_decay": [0.01, 0.001, 0.0001],
    }
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.GridSampler(search_space),
                                )
    study.optimize(objective)
    optuna.visualization.plot_optimization_history(study).write_image("optimization_history.png")


if __name__ == "__main__":
    main()

