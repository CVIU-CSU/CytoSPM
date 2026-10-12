# Copyright (c) Tencent Inc. All rights reserved.
# Modified by the CytoSPM Authors, 2026.
from typing import List, Tuple, Union

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from torch import Tensor

from .yolo_detector import YOLODetector


@MODELS.register_module()
class YOLOWorldDetector(YOLODetector):
    """Open-vocabulary YOLO detector driven by class-name embeddings.

    Args:
        num_train_classes (int): Number of training classes.
    """

    def __init__(self, *args, num_train_classes: int = 80, **kwargs) -> None:
        self.num_train_classes = num_train_classes
        super().__init__(*args, **kwargs)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_train_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        return self.bbox_head.loss(img_feats, txt_feats, batch_data_samples)

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples."""
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        # The vocabulary size is defined by the prompt bank held in the head.
        self.bbox_head.num_classes = \
            self.bbox_head.head_module.cls_contrasts[0].prompt_bank.shape[0]
        results_list = self.bbox_head.predict(img_feats,
                                              txt_feats,
                                              batch_data_samples,
                                              rescale=rescale)
        return self.add_pred_to_datasample(batch_data_samples, results_list)

    def reparameterize(self, texts: List[List[str]]) -> None:
        """Encode a fixed vocabulary once for repeated inference."""
        self.texts = texts
        self.text_feats = self.backbone.forward_text(texts)

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward without post-processing."""
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        return self.bbox_head.forward(img_feats, txt_feats)

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract image features and class-name embeddings."""
        txt_feats = None
        if batch_data_samples is None:
            texts = self.texts
            txt_feats = self.text_feats
        elif isinstance(batch_data_samples,
                        dict) and 'texts' in batch_data_samples:
            texts = batch_data_samples['texts']
        elif isinstance(batch_data_samples, list) and hasattr(
                batch_data_samples[0], 'texts'):
            texts = [data_sample.texts for data_sample in batch_data_samples]
        elif hasattr(self, 'text_feats'):
            texts = self.texts
            txt_feats = self.text_feats
        else:
            raise TypeError('batch_data_samples should be dict or list.')
        if txt_feats is not None:
            img_feats = self.backbone.forward_image(batch_inputs)
        else:
            img_feats, txt_feats = self.backbone(batch_inputs, texts)
        if self.with_neck:
            img_feats = self.neck(img_feats)
        return img_feats, txt_feats
