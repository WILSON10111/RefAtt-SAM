from typing import Tuple, Dict, Union, List
import torch.nn.functional as F
import numpy as np
import torch
import cv2

def mask_iou(pred_label,label):
    '''
    calculate mask iou for pred_label and gt_label
    '''

    pred_label = (pred_label>0)[0].int()
    label = (label>128)[0].int()

    intersection = ((label * pred_label) > 0).sum()
    union = ((label + pred_label) > 0).sum()
    return intersection / (union + 1e-10)

def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode

def boundary_iou(gt, dt, dilation_ratio=0.02):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    device = gt.device
    dt = (dt>0)[0].cpu().byte().numpy()
    gt = (gt>128)[0].cpu().byte().numpy()

    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    boundary_iou = intersection / (union + 1e-10)
    return torch.tensor(boundary_iou).float().to(device)

def to_numpy(input_tensor: torch.Tensor):
    if hasattr(input_tensor, 'detach'):
        input_tensor = input_tensor.detach()
    if hasattr(input_tensor, 'cpu'):
        input_tensor = input_tensor.cpu()
    return input_tensor.numpy()

def calculate_metrics(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> Dict[str, float]:
    """
    Calculate F1 and IoU metrics between predicted mask and ground truth mask.
    """
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.squeeze().cpu().numpy().astype(bool)
    else:
        pred_mask = pred_mask.squeeze().astype(bool)

    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.squeeze().cpu().numpy().astype(bool)
    else:
        gt_mask = gt_mask.squeeze().astype(bool)

    tp = np.logical_and(pred_mask, gt_mask).sum()
    fp = np.logical_and(pred_mask, ~gt_mask).sum()
    fn = np.logical_and(~pred_mask, gt_mask).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 0.0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = 0.0 if union == 0 else intersection / union

    return {'f1': f1, 'iou': iou}

def calculate_mean_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Calculate mean F1 and IoU from a list of metrics dictionaries.
    """
    f1_list = [metrics['f1'] for metrics in metrics_list]
    iou_list = [metrics['iou'] for metrics in metrics_list]

    mean_f1 = sum(f1_list) / len(f1_list) if f1_list else 0
    mean_iou = sum(iou_list) / len(iou_list) if iou_list else 0

    return {'f1': mean_f1, 'iou': mean_iou}


class SamHQIoU:

    def __init__(self):
        self.reset()

    @staticmethod
    def compute_iou(preds: torch.Tensor, target: torch.Tensor):
        assert target.shape[1] == 1, 'only support one mask per image now'
        if (preds.shape[2] != target.shape[2] or preds.shape[3] != target.shape[3]):
            postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
        else:
            postprocess_preds = preds
        iou_list = []
        for i in range(0, len(preds)):
            iou_list.append(mask_iou(postprocess_preds[i], target[i]))
        return iou_list


    @staticmethod
    def compute_boundary_iou(preds: torch.Tensor, target: torch.Tensor):
        assert target.shape[1] == 1, 'only support one mask per image now'
        if (preds.shape[2] != target.shape[2] or preds.shape[3] != target.shape[3]):
            postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
        else:
            postprocess_preds = preds
        biou_list = []
        for i in range(0, len(preds)):
            biou_list.append(boundary_iou(target[i], postprocess_preds[i]))
        return biou_list


    def update(self, label_trues: List[torch.Tensor], label_preds: List[torch.Tensor], index_name: List):
        assert len(label_preds) == len(label_trues)

        for i in range(len(label_trues)):
            if index_name[i] not in self.index_results.keys():
                self.index_results[index_name[i]] = {}

            if label_trues[i].max() <= 1.0:
                label_trues[i] *= 255.0

            curr_iou = self.compute_iou(label_preds[i], label_trues[i])[0]
            if isinstance(curr_iou, torch.Tensor):
                curr_iou = curr_iou.item()
            self.index_results[index_name[i]]['iou'] = curr_iou

            curr_biou = self.compute_boundary_iou(label_preds[i], label_trues[i])[0]
            if isinstance(curr_biou, torch.Tensor):
                curr_biou = curr_biou.item()
            self.index_results[index_name[i]]['biou'] = curr_biou


    def compute(self) -> Tuple[Dict, Dict]:
        results_dict = {
            "Mean Foreground IoU": sum([item['iou'] for item in self.index_results.values()]) / len(self.index_results),
            "Mean Foreground BIoU": sum([item['biou'] for item in self.index_results.values()]) / len(self.index_results),
        }
        return results_dict, self.index_results


    def reset(self):
        self.index_results = {}


class StreamSegMetrics:
    """
    Stream Metrics for Semantic Segmentation Task
    """

    def __init__(self, class_names):
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.reset()

    def update(self,
               label_trues: Union[torch.Tensor, np.ndarray, List[torch.Tensor]],
               label_preds: Union[torch.Tensor, np.ndarray, List[torch.Tensor]],
               index_name: List):
        for masks in [label_preds, label_trues]:
            for i in range(len(masks)):
                if not isinstance(masks[i], np.ndarray):
                    masks[i] = to_numpy(masks[i])
                if len(masks[i].shape) == 2:
                    masks[i] = masks[i][None, None, :]
                if len(masks[i].shape) == 3:
                    masks[i] = masks[i][None, :]
                if len(masks[i].shape) != 4:
                    raise RuntimeError

        for i, (lt, lp) in enumerate(zip(label_trues, label_preds)):
            index_hist = self._fast_hist(lt.flatten(), lp.flatten())
            self.confusion_matrix += index_hist
            self.index_results[index_name[i]] = self.compute(hist=index_hist)[0]


    def _fast_hist(self, label_true: np.ndarray, label_pred: np.ndarray):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask].astype(int),
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def compute(self, hist=None) -> Tuple[Dict, Dict]:
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        if hist is None:
            hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / (hist.sum(axis=1) + 1e-10)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-10)
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

        results_dict = {
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu,
            "Mean Foreground IoU": np.nanmean(iu[1:]) if len(iu) > 1 else 0.0
        }
        for i, class_name in enumerate(self.class_names):
            results_dict[f'{class_name} IoU'] = iu[i]

        return results_dict, self.index_results

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.index_results = {}
