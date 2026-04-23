from os.path import join
from torch.utils.data import DataLoader
from torch.nn import functional as F
import os
import cv2
import torch
from tqdm import tqdm
from data.datasets.hrsid import HRSIDDataset
import argparse
import warnings
warnings.filterwarnings('ignore')
from attnsam_model import SamPredictor
from attnsam_model.modeling import ATTNSAM
from utils.evaluators import calculate_metrics, calculate_mean_metrics
from utils.preprocess_masks import process_masks

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_idx', type=str, default='image_7')
    parser.add_argument('--outdir', type=str, default='hrsid_test')
    args = parser.parse_args()
    return args

def main():
    args = get_arguments()
    images_path = './data/hrsid/ref/image'
    masks_path = './data/hrsid/ref/mask'
    output_path = './output/' + args.outdir

    if not os.path.exists('./outputs/'):
        os.mkdir('./outputs/')

    attnsam(args, images_path, masks_path, output_path)


def attnsam(args, images_path, masks_path, output_path):
    print("\n------------> Segment ")
    ref_image_path = os.path.join(images_path, args.ref_idx + '.png')
    ref_mask_path = os.path.join(masks_path, args.ref_idx + '.png')
    output_path = os.path.join(output_path)
    os.makedirs(output_path, exist_ok=True)
    ref_image = cv2.imread(ref_image_path)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    ref_mask = cv2.imread(ref_mask_path)
    ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)

    print("======> Load ATTNSAM")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    attnsam = ATTNSAM(model_type='vit_l').to(device=device)
    attnsam_state_dict = torch.load('hrsid_1shot.pth', map_location=device)
    attnsam.load_state_dict(attnsam_state_dict)
    predictor = SamPredictor(attnsam)

    print("======> Target feature extraction")
    ref_mask = predictor.set_image(ref_image, ref_mask)
    ref_feat = predictor.features.squeeze().permute(1, 2, 0)
    ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear")
    ref_mask = ref_mask.squeeze()[0]
    target_feat = ref_feat[ref_mask > 0]
    target_embedding = target_feat.mean(0).unsqueeze(0)
    target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
    target_embedding = target_embedding.unsqueeze(0)

    print('======> Start Testing')
    dataset_class = HRSIDDataset
    test_dataset = dataset_class(
        data_dir=join('./data', 'hrsid'), train_flag=False
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, shuffle=False, drop_last=False,
        batch_size=1, num_workers=0,
        collate_fn=test_dataset.collate_fn
    )
    metrics_list = []
    for test_step, batch in enumerate(tqdm(test_dataloader)):
        batch = batch_to_cuda(batch, device)
        test_image_path = batch['image_path'][0]
        test_image = cv2.imread(test_image_path)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        predictor.set_image(test_image)
        test_feat = predictor.features.squeeze()
        C, h, w = test_feat.shape
        test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
        test_feat = test_feat.reshape(C, h * w)
        sim = target_feat @ test_feat
        sim = torch.from_numpy(sim).unsqueeze(0).unsqueeze(0)
        sim = sim.reshape(1, 1, h, w)
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
        sim = predictor.model.postprocess_masks(
            sim,
            input_size=predictor.input_size,
            original_size=predictor.original_size).squeeze()
        sim = (sim - sim.mean()) / torch.std(sim)
        sim = F.interpolate(sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
        attn_sim = sim.sigmoid_().unsqueeze(0).flatten(3)
        attn_sim = attn_sim.to(device)
        if batch['point_coords'][0] is not None:
            # First-step prediction
            point_coords_squeezed = batch['point_coords'][0].squeeze(1)
            point_coords_array = point_coords_squeezed.cpu().numpy()
            point_label_squeezed = batch['point_labels'][0].squeeze(1)
            point_label_array = point_label_squeezed.cpu().numpy()
            predictor.set_image(test_image)
            masks_pred, _, logits, _ = predictor.predict(
                point_coords=point_coords_array,
                point_labels=point_label_array,
                multimask_output=False,
                attn_sim=attn_sim,
                target_embedding=target_embedding
            )
            # Cascaded Post-Processing
            masks_pred, _, logits, _ = predictor.predict(
                point_coords=point_coords_array,
                point_labels=point_label_array,
                mask_input=logits,
                multimask_output=True
            )
            masks_pred = process_masks(masks_pred[0])
        else:
            masks_pred = [torch.zeros(1, 512, 512)]

        metrics = calculate_metrics(masks_pred[0], batch['gt_masks'][0])
        if metrics['iou'] > 0 and metrics['f1'] > 0:
            metrics_list.append(metrics)
        print(f"Image {test_step}: F1 = {metrics['f1']:.2%}, IoU = {metrics['iou']:.2%}")

    mean_metrics = calculate_mean_metrics(metrics_list)
    print(f"Mean Metrics: F1 = {mean_metrics['f1']:.2%}, IoU = {mean_metrics['iou']:.2%}")


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