import copy

import torch
import torch_scatter

# from .chamfer import chamfer_distance
from .loss import _valid_mean
from .transforms import transform_pc
import pytorch3d.transforms as transforms
import pytorch3d.transforms as p3dt
from pytorch3d.loss.chamfer import chamfer_distance

# @torch.no_grad()
# def calc_part_acc(pts, trans1, trans2, rot1, rot2, valids, ret_cd=False):
#     """Compute the `Part Accuracy` in the paper.

#     We compute the per-part chamfer distance, and the distance lower than a
#         threshold will be considered as correct.

#     Args:
#         pts: [B, P, N, 3], model input point cloud to be transformed
#         trans1: [B, P, 3]
#         trans2: [B, P, 3]
#         rot1: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
#         rot2: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
#         valids: [B, P], 1 for input parts, 0 for padded parts
#         ret_cd: whether return chamfer distance

#     Returns:
#         [B], accuracy per data in the batch
#     """
#     B, P = pts.shape[:2]

#     pts1 = transform_pc(trans1, rot1, pts)  # [B, P, N, 3]
#     pts2 = transform_pc(trans2, rot2, pts)

#     pts1 = pts1.flatten(0, 1)  # [B*P, N, 3]
#     pts2 = pts2.flatten(0, 1)
#     dist1, dist2 = chamfer_distance(pts1, pts2)  # [B*P, N]
#     loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
#     loss_per_data = loss_per_data.view(B, P).type_as(pts)

#     # part with CD < `thre` is considered correct
#     thre = 0.01
#     acc = (loss_per_data < thre) & (valids == 1)
#     # the official code is doing avg per-shape acc (not per-part)
#     acc = acc.sum(-1) / (valids == 1).sum(-1)
#     if ret_cd:
#         cd = loss_per_data.sum(-1) / (valids == 1).sum(-1)
#         return acc, cd
#     return acc


from chamferdist import ChamferDistance


@torch.no_grad()
def calc_part_acc(pts, trans1, trans2, rot1, rot2, valids, ret_cd=False):
    """Compute the `Part Accuracy` in the paper.

    We compute the per-part chamfer distance, and the distance lower than a
        threshold will be considered as correct.

    Args:
        pts: [B, P, N, 3], model input point cloud to be transformed
        trans1: [B, P, 3], pred_translation
        trans2: [B, P, 3], gt_translation
        rot1: [B, P, 4], Rotation3D, quat or rmat
        rot2: [B, P, 4], Rotation3D, quat or rmat
        valids: [B, P], 1 for input parts, 0 for padded parts

    Returns:
        [B], accuracy per data in the batch
    """
    chamfer_distance = ChamferDistance()
    B, P = pts.shape[:2]

    pts1 = transform_pc(trans1, rot1, pts)  # [B, P, N, 3]
    pts2 = transform_pc(trans2, rot2, pts)

    pts1 = pts1.flatten(0, 1)  # [B*P, N, 3]
    pts2 = pts2.flatten(0, 1)
    loss_per_data = chamfer_distance(
        pts1,
        pts2,
        bidirectional=True,
        point_reduction="mean",
        batch_reduction=None,
    )  # [B*P, N]
    loss_per_data = loss_per_data.view(B, P).type_as(pts)

    # part with CD < `thre` is considered correct
    thre = 0.01
    acc_per_part = (loss_per_data < thre) & (valids == 1)
    # the official code is doing avg per-shape acc (not per-part)
    acc = acc_per_part.sum(-1) / (valids == 1).sum(-1)
    if ret_cd:
        cd = loss_per_data.sum(-1) / (valids == 1).sum(-1)
        return acc, cd
    return acc


# @torch.no_grad()
# def calc_shape_cd(pts, trans1, trans2, rot1, rot2, valids):
#     chamfer_distance = ChamferDistance()
#     B, P, N, _ = pts.shape

#     valid_mask = valids[..., None, None]  # [B, P, 1, 1]

#     pts = pts.detach().clone()

#     pts = pts.masked_fill(valid_mask == 0, 1e3)

#     pts1 = transform_pc(trans1, rot1, pts)  # [B, P, N, 3]
#     pts2 = transform_pc(trans2, rot2, pts)

#     shape1 = pts1.flatten(1, 2)
#     shape2 = pts2.flatten(1, 2)

#     shape_cd = chamfer_distance(
#         shape1,
#         shape2,
#         bidirectional=True,
#         point_reduction=None,
#         batch_reduction=None
#     )

#     shape_cd = shape_cd.view(B, P, N).mean(-1)
#     shape_cd = _valid_mean(shape_cd, valids)

#     return shape_cd


@torch.no_grad()
def calc_shape_cd(pts, n_pcs, trans1, rot1, gt_pcs, valids):
    chamfer_distance = ChamferDistance()
    num_parts = valids.sum(-1).to(torch.int32)

    shape_cd = []
    for b in range(pts.shape[0]):
        index = 0
        final_pts = []
        for i in range(num_parts[b].item()):
            c_n_pcs = n_pcs[b, i]
            c_pts = pts[b, index : index + c_n_pcs]
            c_trans = trans1[b, i]
            c_rots = rot1[b, i].to_quat()

            c_pts = transforms.quaternion_apply(c_rots, c_pts)
            c_pts = c_pts + c_trans
            final_pts.append(c_pts)
            index += n_pcs[0][i]
        final_pts = torch.cat(final_pts, dim=0)
        gt_pc = gt_pcs[b]
        cd = chamfer_distance(
            final_pts.unsqueeze(0),
            gt_pc.unsqueeze(0),
            bidirectional=True,
            point_reduction="mean",
            batch_reduction=None,
        )
        shape_cd.append(cd)

    shape_cd = torch.stack(shape_cd, dim=0).squeeze(1)

    return shape_cd


@torch.no_grad()
def calc_connectivity_acc(trans, rot, contact_points):
    """Compute the `Connectivity Accuracy` in the paper.

    We transform pre-computed connected point pairs using predicted pose, then
        we compare the distance between them.
    Distance lower than a threshold will be considered as correct.

    Args:
        trans: [B, P, 3]
        rot: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
        contact_points: [B, P, P, 4], pairwise contact matrix.
            First item is 1 --> two parts are connecting, 0 otherwise.
            Last three items are the contacting point coordinate.

    Returns:
        [B], accuracy per data in the batch
    """
    B, P, _ = trans.shape
    thre = 0.01
    # get torch.Tensor of rotation for simplicity
    rot_type = rot.rot_type
    rot = rot.rot

    def get_min_l2_dist(points1, points2, trans1, trans2, rot1, rot2):
        """Compute the min L2 distance between two set of points."""
        # points1/2: [num_contact, num_symmetry, 3]
        # trans/rot: [num_contact, 3/4/(3, 3)]
        points1 = transform_pc(trans1, rot1, points1, rot_type=rot_type)
        points2 = transform_pc(trans2, rot2, points2, rot_type=rot_type)
        dist = ((points1[:, :, None] - points2[:, None, :]) ** 2).sum(-1)
        return dist.min(-1)[0].min(-1)[0]  # [num_contact]

    # find all contact points
    mask = contact_points[..., 0] == 1  # [B, P, P]
    # points1 = contact_points[mask][..., 1:]
    # TODO: more efficient way of getting paired contact points?
    points1, points2, trans1, trans2, rot1, rot2 = [], [], [], [], [], []
    for b in range(B):
        for i in range(P):
            for j in range(P):
                if mask[b, i, j]:
                    points1.append(contact_points[b, i, j, 1:])
                    points2.append(contact_points[b, j, i, 1:])
                    trans1.append(trans[b, i])
                    trans2.append(trans[b, j])
                    rot1.append(rot[b, i])
                    rot2.append(rot[b, j])
    points1 = torch.stack(points1, dim=0)  # [n, 3]
    points2 = torch.stack(points2, dim=0)  # [n, 3]
    # [n, 3/4/(3, 3)], corresponding translation and rotation
    trans1, trans2 = torch.stack(trans1, dim=0), torch.stack(trans2, dim=0)
    rot1, rot2 = torch.stack(rot1, dim=0), torch.stack(rot2, dim=0)
    points1 = torch.stack(get_sym_point_list(points1), dim=1)  # [n, sym, 3]
    points2 = torch.stack(get_sym_point_list(points2), dim=1)  # [n, sym, 3]
    dist = get_min_l2_dist(points1, points2, trans1, trans2, rot1, rot2)
    acc = (dist < thre).sum().float() / float(dist.numel())

    # the official code is doing avg per-contact_point acc (not per-shape)
    # so we tile the `acc` to [B]
    acc = torch.ones(B).type_as(trans) * acc
    return acc


def get_sym_point(point, x, y, z):
    """Get the symmetry point along one or many of xyz axis."""
    point = copy.deepcopy(point)
    if x == 1:
        point[..., 0] = -point[..., 0]
    if y == 1:
        point[..., 1] = -point[..., 1]
    if z == 1:
        point[..., 2] = -point[..., 2]
    return point


def get_sym_point_list(point, sym=None):
    """Get all poissible symmetry point as a list.
    `sym` is a list indicating the symmetry axis of point.
    """
    if sym is None:
        sym = [1, 1, 1]
    else:
        if not isinstance(sym, (list, tuple)):
            sym = sym.tolist()
        sym = [int(i) for i in sym]
    point_list = []
    for x in range(sym[0] + 1):
        for y in range(sym[1] + 1):
            for z in range(sym[2] + 1):
                point_list.append(get_sym_point(point, x, y, z))

    return point_list


@torch.no_grad()
def trans_metrics(trans1, trans2, valids, metric):
    """Evaluation metrics for transformation.

    Metrics used in the NSM paper.

    Args:
        trans1: [B, P, 3]
        trans2: [B, P, 3]
        valids: [B, P], 1 for input parts, 0 for padded parts
        metric: str, 'mse', 'rmse' or 'mae'

    Returns:
        [B], metric per data in the batch
    """
    assert metric in ["mse", "rmse", "mae"]
    if metric == "mse":
        metric_per_data = (trans1 - trans2).pow(2).mean(dim=-1)  # [B, P]
    elif metric == "rmse":
        metric_per_data = (trans1 - trans2).pow(2).mean(dim=-1) ** 0.5
    else:
        metric_per_data = (trans1 - trans2).abs().mean(dim=-1)
    metric_per_data = _valid_mean(metric_per_data, valids)
    return metric_per_data


@torch.no_grad()
def rot_metrics(rot1, rot2, valids, metric):
    """Evaluation metrics for rotation in euler angle (degree) space.

    Metrics used in the NSM paper.

    Args:
        rot1: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
        rot2: [B, P, 4/(3, 3)], Rotation3D, quat or rmat
        valids: [B, P], 1 for input parts, 0 for padded parts
        metric: str, 'mse', 'rmse' or 'mae'

    Returns:
        [B], metric per data in the batch
    """
    assert metric in ["mse", "rmse", "mae"]
    deg1 = rot1.to_euler(to_degree=True)  # [B, P, 3]
    deg2 = rot2.to_euler(to_degree=True)
    diff1 = (deg1 - deg2).abs()
    diff2 = 360.0 - (deg1 - deg2).abs()
    # since euler angle has the discontinuity at 180
    diff = torch.minimum(diff1, diff2)
    if metric == "mse":
        metric_per_data = diff.pow(2).mean(dim=-1)  # [B, P]
    elif metric == "rmse":
        metric_per_data = diff.pow(2).mean(dim=-1) ** 0.5
    else:
        metric_per_data = diff.abs().mean(dim=-1)
    metric_per_data = _valid_mean(metric_per_data, valids)
    return metric_per_data


@torch.no_grad()
def calc_part_acc_weighted(
    pts: torch.Tensor,  # [B, N_sum, 3]
    gt_trans: torch.Tensor,  # [valid_P, 3]
    gt_rots: torch.Tensor,  # [valid_P, 4]
    pred_trans: torch.Tensor,  # [valid_P, 3]
    pred_rots: torch.Tensor,  # [valid_P, 4]
    points_per_part: torch.Tensor,  # [B, P]
    part_valids: torch.Tensor,  # [B, P]
    part_valids_wo_redundancy: torch.Tensor,  # [B, P]
):
    B, P = part_valids.shape
    points_per_valid_part = points_per_part[part_valids]
    gt_trans_point = gt_trans.repeat_interleave(
        points_per_valid_part, dim=0
    )  # (B*N_sum, 3)
    gt_rots_point = gt_rots.repeat_interleave(points_per_valid_part, dim=0)
    pred_trans_point = pred_trans.repeat_interleave(points_per_valid_part, dim=0)
    pred_rots_point = pred_rots.repeat_interleave(points_per_valid_part, dim=0)
    pts_gt = (
        p3dt.quaternion_apply(gt_rots_point, pts.view(-1, 3)) + gt_trans_point
    ).detach()  # (B*N_sum, 3)
    pts_pred = (
        p3dt.quaternion_apply(pred_rots_point, pts.view(-1, 3)) + pred_trans_point
    ).detach()  # (B*N_sum, 3)

    # padding to (valid_P, N_max, 3)
    N_max = points_per_valid_part.max()
    valid_P = points_per_valid_part.shape[0]
    pts_gt_padded = torch.zeros(valid_P, N_max, 3, device=pts.device)
    pts_pred_padded = torch.zeros(valid_P, N_max, 3, device=pts.device)

    # Create row indices
    row_idx = torch.arange(valid_P, device=pts.device).unsqueeze(1).expand(-1, N_max)
    # Create column indices
    col_idx = torch.arange(N_max, device=pts.device).unsqueeze(0).expand(valid_P, -1)
    mask = col_idx < points_per_valid_part.unsqueeze(1)
    source_idx = torch.arange(pts_gt.shape[0], device=pts.device)
    pts_gt_padded[row_idx[mask], col_idx[mask]] = pts_gt[source_idx]
    pts_pred_padded[row_idx[mask], col_idx[mask]] = pts_pred[source_idx]

    # Compute chamfer distance
    shape_cd, _ = chamfer_distance(
        x=pts_gt_padded,
        y=pts_pred_padded,
        x_lengths=points_per_valid_part,
        y_lengths=points_per_valid_part,
        single_directional=False,
        point_reduction="mean",
        batch_reduction=None,
    )  # (valid_P,)

    # Compute part accuracy
    threshold = 0.01
    acc_per_part = (shape_cd < threshold).float()

    # Following way of calculation is used before we added redundancy part
    # When redundancy part is not added, two methods should be equivalent
    # part_offset = torch.cat(
    #     [torch.tensor([0], device=part_valids.device), part_valids.sum(-1).cumsum(0)]
    # )
    # acc_per_data = torch_scatter.segment_csr(
    #     src=acc_per_part,
    #     indptr=part_offset,
    #     reduce="mean",
    # )

    # Recover to object (B, P, P)
    acc_per_part_padded = torch.zeros(B, P, device=part_valids.device)
    acc_per_part_padded[part_valids] = acc_per_part
    acc_per_part_padded[~part_valids_wo_redundancy] = 0.0
    acc_per_data = acc_per_part_padded.sum(-1) / part_valids_wo_redundancy.sum(-1)

    return acc_per_data


@torch.no_grad()
def calc_shape_cd_weighted(
    pts: torch.Tensor,  # [B, N_sum, 3]
    gt_trans: torch.Tensor,  # [valid_P, 3]
    gt_rots: torch.Tensor,  # [valid_P, 4]
    pred_trans: torch.Tensor,  # [valid_P, 3]
    pred_rots: torch.Tensor,  # [valid_P, 4]
    points_per_part: torch.Tensor,  # [B, P]
    part_valids: torch.Tensor,  # [B, P]
    part_valids_wo_redundancy: torch.Tensor,  # [B, P]
):
    B, N_sum, _ = pts.shape
    points_per_valid_part = points_per_part[part_valids]
    gt_trans_point = gt_trans.repeat_interleave(
        points_per_valid_part, dim=0
    )  # (B*N_sum, 3)
    gt_rots_point = gt_rots.repeat_interleave(points_per_valid_part, dim=0)
    pred_trans_point = pred_trans.repeat_interleave(points_per_valid_part, dim=0)
    pred_rots_point = pred_rots.repeat_interleave(points_per_valid_part, dim=0)

    pts_gt = (
        p3dt.quaternion_apply(gt_rots_point, pts.view(-1, 3)) + gt_trans_point
    ).detach()  # (B*N_sum, 3)
    pts_pred = (
        p3dt.quaternion_apply(pred_rots_point, pts.view(-1, 3)) + pred_trans_point
    ).detach()  # (B*N_sum, 3)

    # Handle redundancy parts
    # The logic here is that we do not consider the chamfer distance for the
    # redundant parts, so we set the distance to be a large number
    # and the redundant parts will not contribute to the final distance
    # and we make sure for redundant parts, thier chamfer distance is 0
    if (part_valids_wo_redundancy != part_valids).any():
        redundancy_parts_mask = part_valids & ~part_valids_wo_redundancy  # (B, P)
        redundancy_parts_mask = redundancy_parts_mask[part_valids]  # (valid_P,)
        redundancy_parts_points_mask = redundancy_parts_mask.repeat_interleave(
            points_per_valid_part, dim=0
        )
        pts_gt[redundancy_parts_points_mask] = 1e3
        pts_pred[redundancy_parts_points_mask] = 1e3

    # Back to the original shape
    pts_gt = pts_gt.view(B, N_sum, -1)
    pts_pred = pts_pred.view(B, N_sum, -1)

    shape_cd, _ = chamfer_distance(
        x=pts_gt,
        y=pts_pred,
        single_directional=False,
        point_reduction=None,
        batch_reduction=None,
    )
    shape_cd = shape_cd[0] + shape_cd[1]  # (B, N_sum)
    # mean over parts
    offset = torch.cat(
        [
            torch.zeros((B, 1), device=points_per_part.device),
            points_per_part.cumsum(-1),  # (B, P)
        ],
        dim=-1,
    ).long()
    shape_cd = torch_scatter.segment_csr(
        src=shape_cd,
        indptr=offset,
        reduce="mean",
    )  # (B, P)

    shape_cd = _valid_mean(shape_cd, part_valids_wo_redundancy)

    return shape_cd
