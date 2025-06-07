from dataset import load_train_val_cifar100
from train import TrainLoop
from torch.optim import SGD
from torch.optim import lr_scheduler
from transformers import AutoModelForImageClassification
from torchvision import transforms
from torch import nn
from argparse import ArgumentParser
# import optuna
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
def create_pytorch_lr_scheduler(optimizer, total_steps, base_lr, decay_type, warmup_steps, linear_end=1e-5):
    import math
    import torch.optim.lr_scheduler as lr_scheduler
    
    def lr_lambda(current_step):
        # Handle warmup period
        if current_step < warmup_steps:
            return current_step / warmup_steps
        
        # After warmup, apply decay
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)        
        if decay_type == 'linear':
            return linear_end / base_lr + (1.0 - linear_end / base_lr) * (1.0 - progress)
        elif decay_type == 'cosine':
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            raise ValueError(f'Unknown decay type: {decay_type}')
    return lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    seed_everything(42)


    h_res=448
    l_res=384
    train_trans = transforms.Compose([
            transforms.Resize(h_res),
            transforms.RandomCrop(l_res),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
        ])

    test_trans=transforms.Compose([
            transforms.Resize(h_res),
            transforms.CenterCrop(l_res),
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
        "log_step": 5,
        'batch_size': 32 if not debug else 4,
        'num_workers': 4 if not debug else 0,
        'data_dir': "./data/cifar100",
        'model_dir':'./pretrained/vit-l-16',
        'freeze_feature_extractor': args.freeze_feature_extractor,
        'early_stopping': False,
        }
    # import yaml
    # with open('config.yaml', 'r') as f:
    #     num_processes = yaml.safe_load(f)['num_processes']

    config['steps'] = 10000 if not debug else 10
    config['eval_step'] = config['steps'] // 2 if not debug else 1

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



    # Define the hyperparameters to tune
    # Base learning rate (the one you pass to optimizer)
    base_lr = 1e-2
    weight_decay = 0  # Add weight decay to prevent overfitting

    # Create a fresh model instance for each trial
    model = AutoModelForImageClassification.from_pretrained(
        config['model_dir'],
        num_labels=len(classes),
        ignore_mismatched_sizes=True,
        image_size=l_res,
    )

    if config['freeze_feature_extractor']:
        for name, param in model.named_parameters():
            if not 'classifier' in name:
                param.requires_grad = False

    # Create the optimizer with the suggested hyperparameters
    optimizer = SGD(model.parameters(), lr=base_lr, weight_decay=weight_decay, momentum=0.9)

    warmup_steps = 500
    scheduler = create_pytorch_lr_scheduler(
        optimizer=optimizer,
        total_steps=config['steps'],
        base_lr=base_lr,
        decay_type='cosine',  # 'cosine' is usually better for transformers
        warmup_steps=warmup_steps,
        linear_end=1e-5
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
        'debug':debug,
        'early_stopping':config['early_stopping'],
        'num_classes':len(classes),
    }
    loop = TrainLoop(**train_args)
    print("Training loop object initialized")

    print("Starting training...")
    loop.run_loop()
    
    # search_space = {
    # "lr": [1e-3, 3e-3, 1e-2, 3e-2],
    # "batch_size": [16, 32, 64],
    # "momentum": [0.9, 0.95, 0.99],
    # "weight_decay": [0.01, 0.001, 0.0001],
    # }
    # study = optuna.create_study(direction="maximize",
    #                             sampler=optuna.samplers.GridSampler(search_space),
    #                             )
    # study.optimize(objective)
    # optuna.visualization.plot_optimization_history(study).write_image("optimization_history.png")


if __name__ == "__main__":
    main()

