# Copyright (c) Tencent Inc. All rights reserved.
# Modified by the CytoSPM Authors, 2026.
import copy
import math
import os
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import build_norm_layer
from mmdet.models.utils import filter_scores_and_topk, multi_apply
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import OptConfigType, OptInstanceList
from mmengine.config import ConfigDict
from mmengine.dist import get_dist_info
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor

from .structural_prompt_matching import StructuralPromptMatching
from .utils import gt_instances_preprocess
from .yolov8_head import YOLOv8Head, YOLOv8HeadModule


class StructuralPromptContrastiveHead(BaseModule):
    """Class-aware scoring of region features (Stage II of CytoSPM).

    The score of region ``v`` for class ``c`` combines three terms::

        s(v, c) = <v, t_c> + lambda1 * <v, p_c> + lambda2 * max_j <v, b_j^c>

    ``t_c`` is the class-name embedding, ``p_c`` the image-conditioned
    structural prototype and ``b_j^c`` the selected cytomorphology prompts
    (see :class:`StructuralPromptMatching`). During training ``lambda1`` and
    ``lambda2`` are ramped up linearly over ``warmup_iters`` iterations.

    Args:
        embed_dims (int): Region / text embedding dimension.
        prompt_bank (Tensor): Prompt embeddings ``[C, M, D]``.
        norm_cfg (dict): Normalization layer applied to region features.
        bank_mask (Tensor, optional): ``[C, M]`` mask of valid prompts.
        top_p, top_q, tau, chunk, use_prototype: see
            :class:`StructuralPromptMatching`.
        lambda1 (float): Weight of the structural prototype score.
        lambda2 (float): Weight of the fine-grained attribute score.
        warmup_iters (int): Linear warm-up length of lambda1 / lambda2.
    """

    def __init__(self,
                 embed_dims: int,
                 prompt_bank: Tensor,
                 norm_cfg: ConfigDict,
                 bank_mask: Optional[Tensor] = None,
                 top_p: float = 0.01,
                 top_q: float = 0.0,
                 tau: float = 0.1,
                 chunk: int = 8,
                 use_prototype: bool = True,
                 lambda1: float = 0.1,
                 lambda2: float = 0.05,
                 warmup_iters: int = 0,
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.bias = nn.Parameter(torch.zeros([]))
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))
        self.matching = StructuralPromptMatching(
            top_p=top_p, top_q=top_q, tau=tau, chunk=chunk,
            use_prototype=use_prototype, bank_mask=bank_mask)
        self.use_prototype = bool(use_prototype)
        self.warmup_iters = int(warmup_iters)
        self.register_buffer('lambda1', torch.tensor(float(lambda1)), persistent=False)
        self.register_buffer('lambda2', torch.tensor(float(lambda2)), persistent=False)
        self.register_buffer('_struct_step', torch.zeros([], dtype=torch.long))
        # The prompt bank comes from the config, so a base-trained checkpoint
        # can be evaluated with a novel prompt bank of a different size.
        self.register_buffer(
            'prompt_bank', prompt_bank.float().contiguous(), persistent=False)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Region features ``[B, D, H, W]``.
            w (Tensor): Class-name embeddings ``[B, C, D]``.
        """
        x = self.norm(x)
        s3, s2 = self.matching(x, self.prompt_bank)
        assert w.dim() == 3 and w.shape[1] == s3.shape[1] \
            and w.shape[2] == x.shape[1], (
                f'class-name embeddings {tuple(w.shape)} do not match the '
                f'prompt bank ({s3.shape[1]} classes, {x.shape[1]} dims)')
        wn = F.normalize(w.to(x.dtype), dim=-1, p=2)
        s = torch.einsum('bchw,bkc->bkhw', x, wn)
        ramp = 1.0
        if self.training and self.warmup_iters > 0:
            self._struct_step += 1
            ramp = (self._struct_step.float() / self.warmup_iters).clamp(max=1.0)
        s = s + (ramp * self.lambda2) * s3
        if self.use_prototype and s2 is not None:
            s = s + (ramp * self.lambda1) * s2
        return s * self.logit_scale.exp() + self.bias


@MODELS.register_module()
class YOLOWorldHeadModule(YOLOv8HeadModule):
    """YOLO-World head module with structural prompt matching.

    Args:
        embed_dims (int): Region / text embedding dimension.
        model_size (str): Backbone size, selects the input channels.
        prompt_matching (dict): Structural prompt matching settings:
            ``prompt_bank_path`` (``.pt`` with ``emb`` ``[C, M, D]`` and an
            optional ``mask`` ``[C, M]``) plus the keyword arguments of
            :class:`StructuralPromptContrastiveHead`.
    """

    _IN_CHANNELS = {
        'tiny': [96, 192, 384],
        'small': [96, 192, 384],
        'base': [128, 256, 512],
        'large': [192, 384, 768],
    }

    def __init__(self,
                 *args,
                 embed_dims: int,
                 model_size: str,
                 prompt_matching: dict,
                 **kwargs) -> None:
        self.embed_dims = embed_dims
        self.model_size = model_size
        self.prompt_matching = dict(prompt_matching)
        super().__init__(*args, **kwargs)

    def init_weights(self, prior_prob=0.01):
        super().init_weights()
        for cls_pred, cls_contrast, stride in zip(self.cls_preds,
                                                  self.cls_contrasts,
                                                  self.featmap_strides):
            cls_pred[-1].bias.data[:] = 0.0
            nn.init.constant_(
                cls_contrast.bias.data,
                math.log(5 / self.num_classes / (640 / stride)**2))

    def _load_prompt_bank(self, path: str) -> Tuple[Tensor, Optional[Tensor]]:
        pkg = torch.load(path, map_location='cpu', weights_only=True)
        bank = pkg['emb'] if isinstance(pkg, dict) else pkg
        bank = torch.as_tensor(bank).float()
        assert bank.dim() == 3, f'prompt bank must be [C, M, D], got {tuple(bank.shape)}'
        assert bank.shape[-1] == self.embed_dims, (
            f'prompt bank dim {bank.shape[-1]} != embed_dims {self.embed_dims}')
        mask = None
        if isinstance(pkg, dict) and 'mask' in pkg:
            mask = torch.as_tensor(pkg['mask']).bool()
        return bank, mask

    def _init_layers(self) -> None:
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.cls_contrasts = nn.ModuleList()

        reg_out_channels = max(
            (16, self.in_channels[0] // 4, self.reg_max * 4))
        cls_out_channels = 256
        self.in_channels = self._IN_CHANNELS.get(self.model_size, self.in_channels)

        pm = dict(self.prompt_matching)
        bank, mask = self._load_prompt_bank(pm.pop('prompt_bank_path'))

        for i in range(self.num_levels):
            self.reg_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=reg_out_channels,
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=reg_out_channels,
                              out_channels=4 * self.reg_max,
                              kernel_size=1)))
            self.cls_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=cls_out_channels,
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=cls_out_channels,
                              out_channels=self.embed_dims,
                              kernel_size=1)))
            self.cls_contrasts.append(
                StructuralPromptContrastiveHead(
                    self.embed_dims,
                    bank.clone(),
                    self.norm_cfg,
                    bank_mask=mask.clone() if mask is not None else None,
                    **pm))

        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer('proj', proj, persistent=False)

    def forward(self, img_feats: Tuple[Tensor],
                txt_feats: Tensor) -> Tuple[List]:
        assert len(img_feats) == self.num_levels
        txt_feats = [txt_feats for _ in range(self.num_levels)]
        return multi_apply(self.forward_single, img_feats, txt_feats,
                           self.cls_preds, self.reg_preds, self.cls_contrasts)

    def forward_single(self, img_feat: Tensor, txt_feat: Tensor,
                       cls_pred: nn.ModuleList, reg_pred: nn.ModuleList,
                       cls_contrast: nn.ModuleList) -> Tuple:
        b, _, h, w = img_feat.shape
        cls_embed = cls_pred(img_feat)
        cls_logit = cls_contrast(cls_embed, txt_feat)
        bbox_dist_preds = reg_pred(img_feat)
        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        return cls_logit, bbox_preds


@MODELS.register_module()
class YOLOWorldHead(YOLOv8Head):
    """YOLO-World head with organ-conditional class masking.

    Each PentaCyto image comes from a single organ. When an organ mask
    ``[C, O]`` is given, classes of other organs are excluded from the
    classification loss (``organ_loss_mask=True``) and their scores are set to
    zero at inference. The organ of an image is read from ``organ_id`` in its
    meta information (see ``OrganExtractor``).

    Args:
        world_size (int): Number of GPUs used to scale the losses; -1 reads
            it from the distributed environment.
        organ_loss_mask (bool): Mask cross-organ classes in the loss.
        organ_mask_path (str, optional): ``.pt`` file holding ``mask``
            ``[C, O]``.
    """

    def __init__(self,
                 *args,
                 world_size: int = -1,
                 organ_loss_mask: bool = False,
                 organ_mask_path: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.world_size = world_size
        self.organ_loss_mask_enabled = organ_loss_mask
        if organ_mask_path is not None:
            if not os.path.exists(organ_mask_path):
                raise FileNotFoundError(
                    f'organ_mask_path does not exist: {organ_mask_path}')
            mask_pkg = torch.load(organ_mask_path, weights_only=True,
                                  map_location='cpu')
            self.register_buffer('organ_class_mask', mask_pkg['mask'],
                                 persistent=False)

    def set_organ_class_mask(self, mask: Optional[Tensor]) -> None:
        """Replace (or clear with ``None``) the organ class mask.

        Used when evaluating a vocabulary different from the training one.
        Call it on the unwrapped model, not on a DDP wrapper.
        """
        assert not isinstance(self, nn.parallel.DistributedDataParallel)
        if 'organ_class_mask' in self._buffers:
            del self._buffers['organ_class_mask']
        if mask is not None:
            self.register_buffer('organ_class_mask', mask, persistent=False)

    def _per_image_organ_mask(self, organ_mask: Tensor,
                              batch_img_metas: Sequence[dict]) -> Tensor:
        """Return the ``[B, C]`` in-organ class mask of a batch."""
        organ_ids = []
        for b, meta in enumerate(batch_img_metas):
            o = meta.get('organ_id', -1) if isinstance(meta, dict) else -1
            if isinstance(o, torch.Tensor):
                o = int(o.item())
            if o is None or isinstance(o, bool) or not (
                    0 <= int(o) < organ_mask.shape[1]):
                raise RuntimeError(
                    f'sample {b} has no valid organ_id in its meta info '
                    f'(got {o!r}); add OrganExtractor to the pipeline and '
                    f'"organ_id" to PackDetInputs.meta_keys.')
            organ_ids.append(int(o))
        organ_ids_t = torch.as_tensor(organ_ids, device=organ_mask.device)
        return organ_mask[:, organ_ids_t].T.contiguous()

    def loss(self, img_feats: Tuple[Tensor], txt_feats: Tensor,
             batch_data_samples: dict) -> dict:
        outs = self(img_feats, txt_feats)
        loss_inputs = outs + (batch_data_samples['bboxes_labels'],
                              batch_data_samples['img_metas'])
        return self.loss_by_feat(*loss_inputs)

    def forward(self, img_feats: Tuple[Tensor],
                txt_feats: Tensor) -> Tuple[List]:
        return self.head_module(img_feats, txt_feats)

    def predict(self,
                img_feats: Tuple[Tensor],
                txt_feats: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = False) -> List[InstanceData]:
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self(img_feats, txt_feats)
        return self.predict_by_feat(*outs,
                                    batch_img_metas=batch_img_metas,
                                    rescale=rescale)

    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        num_imgs = len(batch_img_metas)

        current_featmap_sizes = [
            cls_score.shape[2:] for cls_score in cls_scores
        ]
        if current_featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = current_featmap_sizes

            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                self.featmap_sizes_train,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
                with_stride=True)

            self.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.flatten_priors_train = torch.cat(mlvl_priors_with_stride,
                                                  dim=0)
            self.stride_tensor = self.flatten_priors_train[..., [2]]

        # gt info
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]  # xyxy
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pred info
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.num_classes)
            for cls_pred in cls_scores
        ]
        flatten_pred_bboxes = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        # (bs, n, 4 * reg_max)
        flatten_pred_dists = [
            bbox_pred_org.reshape(num_imgs, -1, self.head_module.reg_max * 4)
            for bbox_pred_org in bbox_dist_preds
        ]

        flatten_dist_preds = torch.cat(flatten_pred_dists, dim=1)
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_pred_bboxes = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2], flatten_pred_bboxes,
            self.stride_tensor[..., 0])

        # Organ-conditional class mask [B, C].
        per_image_mask = None
        if self.organ_loss_mask_enabled:
            organ_mask = getattr(self, 'organ_class_mask', None)
            if organ_mask is None:
                raise RuntimeError(
                    'organ_loss_mask=True requires organ_mask_path.')
            if organ_mask.shape[0] != flatten_cls_preds.shape[-1]:
                raise RuntimeError(
                    f'organ_class_mask has {organ_mask.shape[0]} classes but '
                    f'cls_pred has {flatten_cls_preds.shape[-1]}.')
            organ_mask = organ_mask.to(
                device=flatten_cls_preds.device, dtype=flatten_cls_preds.dtype)
            per_image_mask = self._per_image_organ_mask(organ_mask,
                                                        batch_img_metas)

        assigned_result = self.assigner(
            (flatten_pred_bboxes.detach()).type(gt_bboxes.dtype),
            flatten_cls_preds.detach().sigmoid(), self.flatten_priors_train,
            gt_labels, gt_bboxes, pad_bbox_flag)

        assigned_bboxes = assigned_result['assigned_bboxes']
        assigned_scores = assigned_result['assigned_scores']
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']

        assigned_scores_sum = assigned_scores.sum().clamp(min=1)

        loss_cls_elem = self.loss_cls(flatten_cls_preds, assigned_scores)
        if per_image_mask is not None:
            # Zero cross-organ classes and rescale by C / valid_C so images of
            # organs with few classes are not under-weighted.
            B, C = per_image_mask.shape
            valid_C = per_image_mask.sum(dim=1).clamp(min=1.0)
            scale = (C / valid_C).view(B, 1, 1)
            loss_cls_elem = loss_cls_elem * per_image_mask.unsqueeze(1) * scale
        loss_cls = loss_cls_elem.sum()
        loss_cls /= assigned_scores_sum

        # rescale bbox
        assigned_bboxes /= self.stride_tensor
        flatten_pred_bboxes /= self.stride_tensor

        # select positive samples mask
        num_pos = fg_mask_pre_prior.sum()
        if num_pos > 0:
            fg_mask_pre_prior = fg_mask_pre_prior.contiguous()
            prior_bbox_mask = (fg_mask_pre_prior.unsqueeze(-1)
                               .expand(-1, -1, 4).contiguous())
            pred_bboxes_pos = torch.masked_select(
                flatten_pred_bboxes, prior_bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, prior_bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(assigned_scores.sum(-1),
                                              fg_mask_pre_prior).unsqueeze(-1)
            loss_bbox = self.loss_bbox(
                pred_bboxes_pos, assigned_bboxes_pos,
                weight=bbox_weight) / assigned_scores_sum

            # dfl loss
            pred_dist_pos = flatten_dist_preds[fg_mask_pre_prior]
            assigned_ltrb = self.bbox_coder.encode(
                self.flatten_priors_train[..., :2] / self.stride_tensor,
                assigned_bboxes,
                max_dis=self.head_module.reg_max - 1,
                eps=0.01)
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
            loss_dfl = self.loss_dfl(pred_dist_pos.reshape(
                -1, self.head_module.reg_max),
                                     assigned_ltrb_pos.reshape(-1),
                                     weight=bbox_weight.expand(-1,
                                                               4).reshape(-1),
                                     avg_factor=assigned_scores_sum)
        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0
            loss_dfl = flatten_pred_bboxes.sum() * 0
        if self.world_size == -1:
            _, world_size = get_dist_info()
        else:
            world_size = self.world_size
        return dict(loss_cls=loss_cls * num_imgs * world_size,
                    loss_bbox=loss_bbox * num_imgs * world_size,
                    loss_dfl=loss_dfl * num_imgs * world_size)

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = True,
                        with_nms: bool = True) -> List[InstanceData]:
        """Decode head outputs into boxes, scores and labels per image."""
        assert len(cls_scores) == len(bbox_preds)
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        multi_label = cfg.multi_label
        multi_label &= self.num_classes > 1
        cfg.multi_label = multi_label

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        # If the shape does not change, use the previous mlvl_priors
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device)
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full(
                (featmap_size.numel() * self.num_base_priors, ), stride) for
            featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.num_classes)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()

        # Zero the scores of classes from other organs.
        organ_mask = getattr(self, 'organ_class_mask', None)
        if organ_mask is not None:
            if organ_mask.shape[0] != flatten_cls_scores.shape[-1]:
                raise RuntimeError(
                    f'organ_class_mask has {organ_mask.shape[0]} classes but '
                    f'cls_scores has {flatten_cls_scores.shape[-1]}; call '
                    f'set_organ_class_mask() with a matching mask or None.')
            organ_mask = organ_mask.to(
                device=flatten_cls_scores.device,
                dtype=flatten_cls_scores.dtype)
            per_image_mask = self._per_image_organ_mask(organ_mask,
                                                        batch_img_metas)
            flatten_cls_scores = flatten_cls_scores * per_image_mask.unsqueeze(1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_decoded_bboxes = self.bbox_coder.decode(
            flatten_priors[None], flatten_bbox_preds, flatten_stride)

        results_list = []
        for bboxes, scores, img_meta in zip(flatten_decoded_bboxes,
                                            flatten_cls_scores,
                                            batch_img_metas):
            ori_shape = img_meta['ori_shape']
            scale_factor = img_meta['scale_factor']
            pad_param = img_meta.get('pad_param', None)
            score_thr = cfg.get('score_thr', -1)

            if scores.shape[0] == 0:
                empty_results = InstanceData()
                empty_results.bboxes = bboxes
                empty_results.scores = scores[:, 0]
                empty_results.labels = scores[:, 0].int()
                results_list.append(empty_results)
                continue

            nms_pre = cfg.get('nms_pre', 100000)
            if cfg.multi_label is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(labels=labels[:, 0]))
                labels = results['labels']
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(
                    scores, score_thr, nms_pre)

            results = InstanceData(scores=scores,
                                   labels=labels,
                                   bboxes=bboxes[keep_idxs])

            if rescale:
                if pad_param is not None:
                    results.bboxes -= results.bboxes.new_tensor([
                        pad_param[2], pad_param[0], pad_param[2], pad_param[0]
                    ])
                results.bboxes /= results.bboxes.new_tensor(
                    scale_factor).repeat((1, 2))

            results = self._bbox_post_process(results=results,
                                              cfg=cfg,
                                              rescale=False,
                                              with_nms=with_nms,
                                              img_meta=img_meta)
            results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
            results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

            results_list.append(results)
        return results_list
