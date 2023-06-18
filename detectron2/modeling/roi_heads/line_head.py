import numpy as np
from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, cat, nonzero_tuple
from detectron2.modeling.box_regression import Line2LineTransform, _dense_line_regression_loss
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

ROI_LINE_HEAD_REGISTRY = Registry("ROI_LINE_HEAD")
ROI_LINE_HEAD_REGISTRY.__doc__ = """
Registry for line heads, which predicts instance line given
per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""

class BaseLinesRCNNHead(nn.Module):

    @configurable
    def __init__(self, *, loss_weight: float = 1.0):
        """
        NOTE: this interface is experimental.

        Args:
            loss_weight (float): multiplier of the loss
        """
        super().__init__()
        self.loss_weight = loss_weight

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {"loss_weight": cfg.LINE_LOSS_WEIGHT}
    
    def forward(self, x, instances: List[Instances]):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.

        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """
        x = self.layers(x)
        if self.training:
            return {"loss_line": self.losses(x, instances) * self.loss_weight}
        else:
            lines = self.predict_lines(x, instances)

            for line, instance in zip(lines, instances):
                instance.pred_lines = line

            return instances
        
    def layers(self, x):
        """
        Neural network layers that makes predictions from input features.
        """
        raise NotImplementedError
    
    def predict_lines(
        self, predictions: torch.Tensor, proposals: List[Instances]
    ):
        raise NotImplementedError
    
    def losses(
        self, predictions: torch.Tensor, proposals: List[Instances]
    ):
        raise NotImplementedError
    
    def line_reg_loss(self, proposal_boxes, gt_lines, pred_deltas):
        raise NotImplementedError

    
# To get torchscript support, we make the head a subclass of `nn.Sequential`.
# Therefore, to add new layers in this head class, please make sure they are
# added in the order they will be used in forward().
@ROI_LINE_HEAD_REGISTRY.register()
class LineRCNNConvUpsampleHead(BaseLinesRCNNHead, nn.Sequential):
    """
    A standard line head.
    """
    
    @configurable
    def __init__(self, input_shape: ShapeSpec, *, fc_dims: List[int], line2line_transform, **kwargs):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            fc_dims (list[int]): the output dimensions of the fc layers
        """
        super().__init__(**kwargs)
        assert len(fc_dims) > 0

        self.line2line_transform = line2line_transform

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):
            if k == 0:
                self.add_module("flatten", nn.Flatten())
            fc = nn.Linear(int(np.prod(self._output_size)), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.add_module("fc_relu{}".format(k + 1), nn.ReLU())
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

        #add final layer
        self.line_pred = nn.Linear(int(np.prod(self._output_size)), 2)
        self.add_module("line_pred", self.line_pred)
        self.add_module("line_pred_tanh", nn.Tanh())

        nn.init.normal_(self.line_pred.weight, std=0.001)
        for l in [self.line_pred]:
            nn.init.constant_(l.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        num_fc = cfg.MODEL.ROI_LINE_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_LINE_HEAD.FC_DIM

        ret = super().from_config(cfg, input_shape)
        ret["input_shape"] = input_shape
        ret["fc_dims"] = [fc_dim] * num_fc
        ret["line2line_transform"] = Line2LineTransform(weights=cfg.MODEL.ROI_LINE_HEAD.LINE_REG_WEIGHTS)
        return ret

    def layers(self, x):
        for layer in self:
            x = layer(x)
        return x

    def predict_lines(
        self, predictions: torch.Tensor, proposals: List[Instances]
    ): 
        if not len(proposals):
            return []
        
        propossal_deltas = predictions
        
        num_prop_per_image = [len(p) for p in proposals]

        if self.training:
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        else:
            proposal_boxes = cat([p.orig_proposals.tensor for p in proposals], dim=0)
    
        predict_lines = self.line2line_transform.apply_deltas(
            propossal_deltas,
            proposal_boxes,
        )

        return predict_lines.split(num_prop_per_image)

    def losses(
        self, predictions: torch.Tensor, proposals: List[Instances]
    ):
        proposal_deltas = predictions

        # parse line regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            gt_lines = torch.as_tensor(np.concatenate([(p.gt_lines.lines) for p in proposals], axis=0), 
                                        dtype=torch.float32, device=proposal_deltas.device)
        else:
            proposal_boxes = torch.empty((0, 4), device=proposal_deltas.device)
            gt_lines = torch.empty((0, 2), device=proposal_deltas.device)


        return self.line_reg_loss(proposal_boxes, gt_lines, proposal_deltas)

    def line_reg_loss(self, proposal_boxes, gt_lines, pred_deltas):
        loss_line_reg = _dense_line_regression_loss(
            [proposal_boxes],
            self.line2line_transform,
            [pred_deltas.unsqueeze(0)],
            [gt_lines],
        )

        return loss_line_reg



def build_line_head(cfg, input_shape):
    """
    Build a  line defined by `cfg.MODEL.ROI_LINE_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_LINE_HEAD.NAME
    return ROI_LINE_HEAD_REGISTRY.get(name)(cfg, input_shape)
    