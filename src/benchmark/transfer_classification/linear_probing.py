import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import random
from pathlib import Path
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision.transforms import Normalize
from torch.utils import data

from torchgeo.models.vit import ViTLarge16_Weights, vit_large_patch16_224
from torchgeo.datasets.eurosat import EuroSAT
from classification_trainer import ClassificationTrainer
from classification_tester import ClassificationTester


def get_args_parser():
    parser = argparse.ArgumentParser('MAE linear probing for image classification', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lr', type=float, default=None, metavar='LR')

    # * Finetuning params
    parser.add_argument('--cls_token', action='store_false', dest='global_pool')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--nb_classes', default=10, type=int)

    parser.add_argument('--output_dir', default='./output_dir')
    parser.add_argument('--log_dir', default='./log_dir')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    return parser


# Band statistics for normalization
BAND_STATS = {
    'mean': [
        1353.72696296, 1117.20222222, 1041.8842963, 946.554, 1199.18896296,
        2003.00696296, 2374.00874074, 2301.22014815, 732.18207407, 12.09952894,
        1820.69659259, 1118.20259259, 2599.78311111,
    ],
    'std': [
        897.27143653, 736.01759721, 684.77615743, 620.02902871, 791.86263829,
        1341.28018273, 1595.39989386, 1545.52915718, 475.11595216, 98.26600935,
        1216.48651476, 736.6981037, 1750.12066835,
    ]
}


# Wrapper so torchvision transforms apply only to dict["image"]
class DictTransform:
    def __init__(self, tfm):
        self.tfm = tfm

    def __call__(self, sample):
        sample["image"] = self.tfm(sample["image"])
        return sample


# Collate function to turn list of dicts -> (images, labels)
def collate_fn(batch):
    images = [sample["image"] for sample in batch]
    labels = [sample["label"] for sample in batch]
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels)
    return images, labels


def main(args):
    # Set up device and seed
    device = torch.device(args.device)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.benchmark = True  # comment if issues on MPS

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = vit_large_patch16_224(weights=ViTLarge16_Weights.SENTINEL2_ALL_MAE)
    model.head = nn.Linear(model.head.in_features, args.nb_classes)
    model.head = torch.nn.Sequential(
        torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
        model.head
    )

    # Initialize head parameters
    for m in model.head.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # Freeze all except head
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    model.to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    # Transforms
    transform_train = DictTransform(transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=BAND_STATS['mean'], std=BAND_STATS['std']),
    ]))
    transform_val = DictTransform(transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=BAND_STATS['mean'], std=BAND_STATS['std']),
    ]))

    # Datasets
    train_dataset = EuroSAT(root=args.data_path, split='train', transforms=transform_train, download=True)
    val_dataset   = EuroSAT(root=args.data_path, split='val',   transforms=transform_val,   download=True)
    test_dataset  = EuroSAT(root=args.data_path, split='test',  transforms=transform_val,   download=True)

    # Loaders (with collate_fn!)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, pin_memory=args.pin_mem,
                                   collate_fn=collate_fn)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=args.pin_mem,
                                 collate_fn=collate_fn)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=args.pin_mem,
                                  collate_fn=collate_fn)

    # Train
    ClassificationTrainer(
        model=model,
        train_dl=train_loader,
        val_dl=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        saved_models_dir=args.output_dir,
        num_epochs=args.epochs,
        patience=10,
        log_dir=args.log_dir,
        start_epoch=args.start_epoch
    ).run()

    # Test
    ClassificationTester(
        model=model,
        test_dl=test_loader,
        device=device
    ).run()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
