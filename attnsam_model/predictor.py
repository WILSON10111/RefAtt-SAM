# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

from .utils.transforms import ResizeLongestSide


class SamPredictor:
    def __init__(
            self,
            sam_model,
    ) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam): The model to use for mask prediction.
        """
        super().__init__()
        self.model = sam_model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.reset_image()

    def set_image(
        self,
        image: np.ndarray,
        mask: np.ndarray = None,
        image_format: str = "RGB",
        cal_image=True
    ) -> None:
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        # Transform the mask to the form expected by the model
        input_mask_torch = None
        if mask is not None:
          input_mask = self.transform.apply_image(mask)
          input_mask_torch = torch.as_tensor(input_mask, device=self.device)
          input_mask_torch = input_mask_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        input_mask = self.set_torch_image(input_image_torch, image.shape[:2], transformed_mask=input_mask_torch)
        return input_mask

    @torch.no_grad()
    def set_torch_image(
            self,
            transformed_image: torch.Tensor,
            original_image_size: Tuple[int, ...],
            transformed_mask: torch.Tensor = None,
            cal_image=True
    ) -> None:
        """
               计算提供的图像的嵌入，允许使用'predict'方法预测掩码。期望输入图像已经是模型预期的格式。

               参数：
               transformed_image (torch.Tensor): 转换后的输入图像，形状为1x3xHxW，已经通过ResizeLongestSide转换。
               original_image_size (tuple(int, int)): 图像转换前的尺寸，格式为(H, W)。
               transformed_mask (torch.Tensor): 转换后的输入掩码。
               cal_image (bool): 是否计算图像嵌入。
        """
        assert (
                len(transformed_image.shape) == 4
                and transformed_image.shape[1] == 3
                and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."

        if cal_image:
            self.reset_image()
            self.original_size = original_image_size
            self.input_size = tuple(transformed_image.shape[-2:])
            input_image = self.model.preprocess1(transformed_image)
            self.image_embeddings, self.interm_embeddings = self.model.image_encoder(input_image)
            self.features, _ = self.model.image_encoder(input_image)
            self.is_image_set = True

        if transformed_mask is not None:
            input_mask = self.model.preprocess1(transformed_mask)
            return input_mask

    def predict(
            self,
            point_coords: Optional[np.ndarray] = None,
            point_labels: Optional[np.ndarray] = None,
            box: Optional[np.ndarray] = None,
            mask_input: Optional[np.ndarray] = None,
            multimask_output: bool = False,
            return_logits: bool = False,
            attn_sim=None,
            target_embedding=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # Transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert (
                    point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]
            mask_input_torch = mask_input_torch.squeeze(0)
        masks, iou_predictions, low_res_masks, high_res_masks = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output,
            return_logits=return_logits,
            attn_sim=attn_sim,
            target_embedding=target_embedding,
        )

        low_res_masks = low_res_masks[0].detach().cpu().numpy()


        return masks, iou_predictions, low_res_masks, high_res_masks

    @torch.no_grad()
    def predict_torch(
            self,
            point_coords: Optional[torch.Tensor],
            point_labels: Optional[torch.Tensor],
            boxes: Optional[torch.Tensor] = None,
            mask_input: Optional[torch.Tensor] = None,
            multimask_output: bool = False,
            return_logits: bool = False,
            attn_sim=None,
            target_embedding=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        # Embed prompts

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        # Predict masks
        batch_size = len(self.features)

        masks_hq, iou_predictions,low_res_masks = self.model.mask_decoder(
            image_embeddings=self.image_embeddings,
            interm_embeddings=self.interm_embeddings,
            image_pe=[self.model.prompt_encoder.get_dense_pe() for _ in range(batch_size)],
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            attn_sim=attn_sim,
            target_embedding=target_embedding,
        )

        # Upscale the masks to the original image resolution
        #high_res_masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)
        high_res_masks = masks_hq
        # if not return_logits:
        #     masks = high_res_masks > self.model.mask_threshold  # 0.0
        #     return masks, iou_predictions, low_res_masks, high_res_masks
        # else:
        postprocess_masks_hq = [m_hq.clone() for m_hq in masks_hq]
        for i in range(len(postprocess_masks_hq)):
         postprocess_masks_hq[i]= self.postprocess(output_masks=postprocess_masks_hq[i], ori_img_size=self.original_size[i])
        return postprocess_masks_hq, iou_predictions, low_res_masks, high_res_masks

    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert self.features is not None, "Features must exist if an image has been set."
        return self.features

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None

    @staticmethod
    def postprocess(output_masks: torch.Tensor, ori_img_size: Tuple):
        # rescale the mask size back to original image size
        output_mask_size = (output_masks.size(-2), output_masks.size(-1))
        if output_mask_size != ori_img_size:
            if len(output_masks.shape) == 3:
                output_masks = output_masks.unsqueeze(1)
            # bilinear will produce non-deterministic gradients during training. For exact reproduction, please
            # change the mode from bilinear to nearest
            output_masks = F.interpolate(
                output_masks, ori_img_size, mode="bilinear", align_corners=False,
            )
        return output_masks

