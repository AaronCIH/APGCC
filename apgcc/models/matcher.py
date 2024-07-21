# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Mostly copy-paste from DETR (https://github.com/facebookresearch/detr).
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

class HungarianMatcher_Crowd(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_point: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the foreground object
            cost_point: This is the relative weight of the L1 error of the points coordinates in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        assert cost_class != 0 or cost_point != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "points": Tensor of dim [batch_size, num_queries, 2] with the predicted point coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_points] (where num_target_points is the number of ground-truth
                           objects in the target) containing the class labels
                 "points": Tensor of dim [num_target_points, 2] containing the target point coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_points)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2] # num_queries = H/8*W/8*lines*row = feature_map*sample_points

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_points = outputs["pred_points"].flatten(0, 1)  # [batch_size * num_queries, 2]

        # Also concat the target labels and points
        # tgt_ids = torch.cat([v["labels"] for v in targets])
        # tgt_idds = torch.cat([v["image_id"] for v in targets])
        tgt_ids = torch.cat([v["labels"] for v in targets])   # torch.ones(gt.shape[0]) 
        tgt_points = torch.cat([v["point"] for v in targets]) # coords in gt map

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        # cost_class.size = [bs*num_queries, num_all_gt_points]
        cost_class = -out_prob[:, tgt_ids]  # means confidence of pred, all_gt_points share same confidence.
        # cost_calss : out_prob(32*1028, 2) -> (32*1028, len(tgt_ids))

        # Compute the L2 cost between point
        # cost_point.size = [bs*num_queries, num_all_gt_points]
        cost_point = torch.cdist(out_points, tgt_points, p=2) 
        # Compute the giou cost between point

        # Final cost matrix, default: w_cost_point = 0.05, w_cost_class=1
        C = self.cost_point * cost_point + self.cost_class * cost_class # compute each predict point to every gt(all batch) scores.

        # split to according patch predict. only want to consider within target points scores.
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v["point"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]  # linear_sum_assignment find the best match.
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]  

def build_matcher_crowd(cfg):
    return HungarianMatcher_Crowd(cost_class=cfg.MATCHER.SET_COST_CLASS, cost_point=cfg.MATCHER.SET_COST_POINT)
