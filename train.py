import numpy as np
from torch.utils.data import DataLoader
import random
from contextlib import nullcontext
from functools import partial
from os.path import join
from transforms import HorizontalFlip, VerticalFlip, RandomCrop
from torch.nn import functional as F
import torch.distributed as dist
import torch
import os
from tqdm import tqdm
from data.datasets.hrsid import HRSIDDataset
from misc import set_randomness
import argparse
import warnings
warnings.filterwarnings('ignore')
from attnsam_model import SamPredictor
from attnsam_model.modeling import ATTNSAM


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='hrsid_train')
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    output_path = './outputs/' + args.outdir

    if not os.path.exists('./outputs/'):
        os.mkdir('./outputs/')
    attnsam_train(args,output_path)

def attnsam_train(args, output_path):
    output_path = os.path.join(output_path)
    os.makedirs(output_path, exist_ok=True)

    print("======> Load ATTNSAM")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    attnsam = ATTNSAM(model_type='vit_l').to(device=device)
    predictor = SamPredictor(attnsam)

    print("======> Start Training")
    set_randomness()
    worker_id = 0
    gpu_num = 1
    base_rank = os.environ['RANK'] if 'RANK' in os.environ.keys() else 0
    local_rank = base_rank * gpu_num + worker_id
    device = torch.device(f"cuda:{worker_id}")
    torch.cuda.set_device(device)
    transforms = [VerticalFlip(p=0.5), HorizontalFlip(p=0.5),
                  RandomCrop(scale=[0.1, 1.0], p=1.0)]
    dataset_class = HRSIDDataset
    max_object_num = 25
    dataset_dir = join('./data', 'hrsid')
    train_dataset = dataset_class(
        data_dir=dataset_dir, train_flag=True, shot_num=1,
        transforms=transforms, max_object_num=max_object_num
    )
    train_bs = 1
    train_workers, val_workers = 4, 2
    sampler = None
    if torch.distributed.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_bs = int(train_bs / torch.distributed.get_world_size())
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=train_bs, shuffle=sampler is None, num_workers=train_workers,
        sampler=sampler, drop_last=False, collate_fn=train_dataset.collate_fn,
        worker_init_fn=partial(worker_init_fn, base_seed=3407)
    )
    optimizer = torch.optim.AdamW(
        params=[p for p in predictor.model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4
    )

    max_epoch_num, valid_per_epochs = 150, 150
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=max_epoch_num, eta_min=1e-5
    )
    os.makedirs(output_path, exist_ok=True)
    predictor.model.train()
    for epoch in range(1, max_epoch_num + 1):
        if hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)

        train_pbar = None
        if local_rank == 0:
            train_pbar = tqdm(total=len(train_dataloader), desc='train', leave=False)
        for train_step, batch in enumerate(train_dataloader):
            batch = batch_to_cuda(batch, device)

            predictor.model.set_infer_img(batch['images'])
            masks_pred = predictor.model(
                imgs=batch['images'],
                point_coords=batch['point_coords'],
                point_labels=batch['point_labels'],
                box_coords=batch['box_coords'],
                noisy_masks=batch['noisy_object_masks'],
            )

            masks_gt = batch['object_masks']

            for masks in [masks_pred, masks_gt]:
                for i in range(len(masks)):
                    if len(masks[i].shape) == 2:
                        masks[i] = masks[i][None, None, :]
                    if len(masks[i].shape) == 3:
                        masks[i] = masks[i][:, None, :]
                    if len(masks[i].shape) != 4:
                        raise RuntimeError

            bce_loss_list, dice_loss_list, focal_loss_list = [], [], []

            for i in range(len(masks_pred)):
                pred, label = masks_pred[i], masks_gt[i]
                label = torch.where(torch.gt(label, 0.), 1., 0.)
                b_loss = F.binary_cross_entropy_with_logits(pred, label.float())
                d_loss = calculate_dice_loss(pred, label)

                bce_loss_list.append(b_loss)
                dice_loss_list.append(d_loss)

            bce_loss = sum(bce_loss_list) / len(bce_loss_list)
            dice_loss = sum(dice_loss_list) / len(dice_loss_list)
            total_loss = bce_loss + dice_loss
            loss_dict = dict(
                total_loss=total_loss.clone().detach(),
                bce_loss=bce_loss.clone().detach(),
                dice_loss=dice_loss.clone().detach()
            )

            backward_context = nullcontext
            if torch.distributed.is_initialized():
                backward_context = predictor.model.no_sync

            with backward_context():
                total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if torch.distributed.is_initialized():
                for key in loss_dict.keys():
                    if hasattr(loss_dict[key], 'detach'):
                        loss_dict[key] = loss_dict[key].detach()
                    torch.distributed.reduce(loss_dict[key], dst=0, op=torch.distributed.ReduceOp.SUM)
                    loss_dict[key] /= torch.distributed.get_world_size()

            if train_pbar:
                train_pbar.update(1)
                str_step_info = "Epoch: {epoch}/{epochs:4}. " \
                                "Loss: {total_loss:.4f}(total), {bce_loss:.4f}(bce), {dice_loss:.4f}(dice)".format(
                    epoch=epoch, epochs=max_epoch_num,
                    total_loss=loss_dict['total_loss'], bce_loss=loss_dict['bce_loss'], dice_loss=loss_dict['dice_loss']
                )
                train_pbar.set_postfix_str(str_step_info)
        scheduler.step()
        if train_pbar:
            train_pbar.clear()

    torch.save(
                predictor.model.state_dict() if not hasattr(predictor.model, 'module') else predictor.model.module.state_dict(),
                join(output_path, "best_model_hrsid.pth")
                )


def calculate_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    assert inputs.size(0) == targets.size(0)
    inputs = inputs.sigmoid()
    inputs, targets = inputs.flatten(1), targets.flatten(1)

    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()


def worker_init_fn(worker_id: int, base_seed: int, same_worker_seed: bool = True):
    seed = base_seed if same_worker_seed else base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def batch_to_cuda(batch, device):
    for key in batch.keys():
        if key in ['images', 'gt_masks', 'point_coords', 'box_coords', 'noisy_object_masks', 'object_masks']:
            batch[key] = [
                item.to(device=device, dtype=torch.float32) if item is not None else None for item in batch[key]
            ]
        elif key in ['point_labels']:
            batch[key] = [
                item.to(device=device, dtype=torch.long) if item is not None else None for item in batch[key]
            ]
    return batch


if __name__ == "__main__":
    main()
