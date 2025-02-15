import os
import pickle

import numpy as np
import pytorch_lightning
import torch
from scipy.spatial.transform import Rotation as R
from torch import optim
import pytorch3d.transforms as p3dt
import json
from utils import (
    Rotation3D,
    trans_metrics,
    rot_metrics,
    calc_part_acc,
    calc_part_acc_weighted,
    calc_shape_cd_weighted,
)
from utils import filter_wd_parameters, CosineAnnealingWarmupRestarts
from utils import estimate_global_transform, dict_to_numpy


class MatchingBaseModel(pytorch_lightning.LightningModule):
    def __init__(self, cfg):
        super(MatchingBaseModel, self).__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self._setup()
        self.test_results = None
        if len(cfg.STATS):
            os.makedirs(cfg.STATS, exist_ok=True)
            self.stats = dict()
            self.stats["datas"] = []
            self.stats["preds"] = []
            self.stats["metrics"] = []
        else:
            self.stats = None

    def _setup(self):
        self.max_num_part = self.cfg.DATA.MAX_NUM_PART

        self.pc_feat_dim = self.cfg.MODEL.PC_FEAT_DIM

    # The flow for this base model is:
    # training_step -> forward_pass -> loss_function ->
    # _loss_function -> forward

    def forward(self, data_dict):
        """Forward pass to predict matching."""
        raise NotImplementedError("forward function should be implemented per model")

    def training_step(self, data_dict, batch_idx, optimizer_idx=-1):
        loss_dict = self.forward_pass(
            data_dict, mode="train", optimizer_idx=optimizer_idx
        )
        return loss_dict["loss"]

    def validation_step(self, data_dict, batch_idx):
        loss_dict = self.forward_pass(data_dict, mode="val", optimizer_idx=-1)
        return loss_dict

    def validation_epoch_end(self, outputs):
        # avg_loss among all data
        # we need to consider different batch_size

        func = (
            torch.tensor if isinstance(outputs[0]["batch_size"], int) else torch.stack
        )
        batch_sizes = func([output.pop("batch_size") for output in outputs]).type_as(
            outputs[0]["loss"]
        )  # [num_batches]
        losses = {
            f"val/{k}": torch.stack([output[k] for output in outputs]).reshape(-1)
            for k in outputs[0].keys()
        }  # each is [num_batches], stacked avg loss in each batch
        avg_loss = {
            k: (v * batch_sizes).sum() / batch_sizes.sum() for k, v in losses.items()
        }
        self.log_dict(avg_loss, sync_dist=True)

    def test_step(self, data_dict, batch_idx):
        loss_dict = self.forward_pass(data_dict, mode="test", optimizer_idx=-1)
        return loss_dict

    def test_epoch_end(self, outputs):
        # avg_loss among all data
        # we need to consider different batch_size
        if isinstance(outputs[0]["batch_size"], int):
            func_bs = torch.tensor
            func_loss = torch.stack
        else:
            func_bs = torch.cat
            func_loss = torch.cat
        batch_sizes = func_bs([output.pop("batch_size") for output in outputs]).type_as(
            outputs[0]["loss"]
        )  # [num_batches]
        losses = {
            f"test/{k}": func_loss([output[k] for output in outputs])
            for k in outputs[0].keys()
        }  # each is [num_batches], stacked avg loss in each batch
        avg_loss = {
            k: (v * batch_sizes).sum() / batch_sizes.sum() for k, v in losses.items()
        }
        print("; ".join([f"{k}: {v.item():.6f}" for k, v in avg_loss.items()]))
        # this is a hack to get results outside `Trainer.test()` function
        self.test_results = avg_loss
        self.log_dict(avg_loss, sync_dist=True)
        if self.cfg.STATS is not None:
            with open(os.path.join(self.cfg.STATS, "saved_stats.pk"), "wb") as f:
                pickle.dump(self.stats, f)

    @torch.no_grad()
    def calc_metric(self, data_dict, trans_dict):
        """
        :param data_dict: must include:
            part_pcs: [B, P, 3]
            part_quat or part_rot: [B, P, 4] or [B, P, 3, 3], the ground truth quaternion or rotation
            part_trans: [B, P, 3], the ground truth transformation
            part_valids: [B, P], 1 for valid part, 0 for padding
        :param trans_dict: must include:
            rot: predicted rotation
            trans: predicted transformation
        :return: metric: will include
            6 eval metric, already the mean of the batch (total / (B*P_valid))
        """
        if "part_rot" not in data_dict:
            part_quat = data_dict.pop("part_quat")
            data_dict["part_rot"] = Rotation3D(part_quat, rot_type="quat").convert(
                "rmat"
            )
        part_valids = data_dict["part_valids"]
        metric_dict = dict()
        part_pcs = data_dict["part_pcs"]
        pred_trans = torch.tensor(
            trans_dict["trans"], dtype=torch.float32, device=part_pcs.device
        )
        pred_rot = torch.tensor(
            trans_dict["rot"], dtype=torch.float32, device=part_pcs.device
        )
        pred_rot = Rotation3D(pred_rot, rot_type="rmat")
        gt_trans, gt_rot = data_dict["part_trans"], data_dict["part_rot"]
        N_SUM = part_pcs.shape[1]
        n_pcs = data_dict["n_pcs"]
        B, P = n_pcs.shape
        # resample part_pcs with same number of points per part to fit the input requirement of chamfer distance
        # part_pcs_resampled = []
        # for b in range(B):
        #     point_sum = 0
        #     new_pcs = []
        #     for p in range(P):
        #         if n_pcs[b, p].item() == 0:
        #             idx = torch.randint(
        #                 low=point_sum - 1, high=point_sum, size=(N_SUM,)
        #             )
        #         else:
        #             idx = torch.randint(
        #                 low=point_sum,
        #                 high=point_sum + n_pcs[b, p].item(),
        #                 size=(N_SUM,),
        #             )
        #         new_pcs.append(part_pcs[b, idx, :])
        #         point_sum += n_pcs[b, p]
        #     new_pcs = torch.stack(new_pcs)
        #     part_pcs_resampled.append(new_pcs)
        # part_pcs_resampled = torch.stack(part_pcs_resampled).to(part_pcs.device)
        # part_acc, cd = calc_part_acc(
        #     part_pcs_resampled,
        #     pred_trans,
        #     gt_trans,
        #     pred_rot,
        #     gt_rot,
        #     part_valids,
        #     ret_cd=True,
        # )

        # When calculating the metrics, we should take care of the redundant parts.
        # The logic here is that, we only consider the valid parts for the metrics calculation.
        # i.e. Only measure how much will the result be affacted by the redundant parts.
        num_parts_wo_redundancy = (
            data_dict["num_parts"] - data_dict["redundancy"]
        )  # (B,)
        # to B, P like part_valids
        part_valids_wo_redundancy = (
            torch.cumsum(part_valids.bool(), dim=-1) <= num_parts_wo_redundancy[:, None]
        ) & part_valids.bool()

        part_acc = calc_part_acc_weighted(
            pts=part_pcs,
            gt_trans=gt_trans[part_valids.bool()],
            gt_rots=gt_rot.to_quat()[part_valids.bool()],
            pred_trans=pred_trans[part_valids.bool()],
            pred_rots=pred_rot.to_quat()[part_valids.bool()],
            points_per_part=n_pcs,
            part_valids=part_valids.bool(),
            part_valids_wo_redundancy=part_valids_wo_redundancy,
        )
        shape_cd = calc_shape_cd_weighted(
            pts=part_pcs,
            gt_trans=gt_trans[part_valids.bool()],
            gt_rots=gt_rot.to_quat()[part_valids.bool()],
            pred_trans=pred_trans[part_valids.bool()],
            pred_rots=pred_rot.to_quat()[part_valids.bool()],
            points_per_part=n_pcs,
            part_valids=part_valids.bool(),
            part_valids_wo_redundancy=part_valids_wo_redundancy,
        )

        raw_metric_dict = dict()
        raw_metric_dict["part_acc"] = part_acc
        raw_metric_dict["chamfer_distance"] = shape_cd
        metric_dict["part_acc"] = part_acc.mean()
        metric_dict["chamfer_distance"] = shape_cd.mean()

        for metric in ["mse", "rmse", "mae"]:
            trans_met = trans_metrics(
                pred_trans, gt_trans, valids=part_valids, metric=metric
            )
            metric_dict[f"trans_{metric}"] = trans_met.mean()
            raw_metric_dict[f"trans_{metric}"] = trans_met
            rot_met = rot_metrics(pred_rot, gt_rot, valids=part_valids, metric=metric)
            metric_dict[f"rot_{metric}"] = rot_met.mean()
            raw_metric_dict[f"rot_{metric}"] = rot_met

        if self.stats is not None:
            # necessary info to restore this test, while making the stats file not too large
            saved_data = {
                "gt_trans": gt_trans,
                "gt_rot": gt_rot,
                "data_id": data_dict["data_id"],
            }
            self.stats["datas"].append(saved_data)
            self.stats["metrics"].append(dict_to_numpy(metric_dict))
            self.stats["preds"].append(dict_to_numpy(trans_dict))

        return metric_dict, raw_metric_dict

    def _loss_function(self, data_dict, out_dict={}, optimizer_idx=-1):
        raise NotImplementedError("loss_function should be implemented per model")

    def global_alignment(self, data_dict, out_dict):
        perm_mat = out_dict["perm_mat"].cpu().numpy()  # [B, N_, N_]
        gt_pcs = data_dict["gt_pcs"].cpu().numpy()
        part_pcs = data_dict["part_pcs"].cpu().numpy()
        part_quat = data_dict["part_quat"].cpu().numpy()
        part_trans = data_dict["part_trans"].cpu().numpy()
        n_pcs = data_dict.get("n_pcs", None)
        if n_pcs is not None:
            n_pcs = n_pcs.cpu().numpy()

        part_valids = data_dict.get("part_valids", None)
        if part_valids is not None:
            part_valids = part_valids.cpu().numpy()
            n_valid = np.sum(part_valids, axis=1, dtype=np.int32)  # [B]
        else:
            n_valid = None

        gt_pcs = gt_pcs[:, :, :3]
        part_pcs = part_pcs[:, :, :3]
        assert n_pcs is not None
        assert part_valids is not None
        assert n_valid is not None

        critical_pcs_idx = data_dict.get("critical_pcs_idx", None)  # [B, N_sum]
        n_critical_pcs = data_dict.get("n_critical_pcs", None)  # [B, P]

        n_critical_pcs = n_critical_pcs.cpu().numpy()
        critical_pcs_idx = critical_pcs_idx.cpu().numpy()

        dataset = self.trainer.test_dataloaders[0].dataset
        matching_dir_name = f"./matching_data/{dataset.category}_vol"
        if dataset.num_removal > 0:
            matching_dir_name += f"_missing_{dataset.num_removal}"
        if dataset.num_redundancy > 0:
            matching_dir_name += f"_redundant_{dataset.num_redundancy}"
        os.makedirs(matching_dir_name, exist_ok=True)

        pred_dict = estimate_global_transform(
            perm_mat,
            part_pcs,
            n_valid,
            n_pcs,
            n_critical_pcs,
            critical_pcs_idx,
            part_quat,
            part_trans,
            align_pivot=True,
            redundancy=data_dict["redundancy"],
            pfpp_matching_data_path=matching_dir_name,
            gt_pcs=gt_pcs,
            data_id=data_dict["data_id"],
        )
        return pred_dict

    def loss_function(self, data_dict, optimizer_idx, mode):
        # loss_dict = None
        out_dict = self.forward(data_dict)

        loss_dict = self._loss_function(data_dict, out_dict, optimizer_idx)

        if "loss" not in loss_dict:
            # if loss is composed of different losses, should combine them together
            # each part should be of shape [B, ] or [int]
            total_loss = 0.0
            for k, v in loss_dict.items():
                if k.endswith("_loss"):
                    total_loss += v * eval(f"self.cfg.LOSS.{k.upper()}_W")
            loss_dict["loss"] = total_loss

        total_loss = loss_dict["loss"]
        if total_loss.numel() != 1:
            loss_dict["loss"] = total_loss.mean()

        # log the batch_size for avg_loss computation
        if not self.training:
            if "batch_size" not in loss_dict:
                loss_dict["batch_size"] = out_dict["batch_size"]

        B = data_dict["part_pcs"].shape[0]
        part_valids = data_dict["part_valids"].bool()
        part_valids_cpu = part_valids.cpu().numpy()

        if mode == "test":
            pred_dict = self.global_alignment(data_dict, out_dict)
            metric_dict, raw_metric_dict = self.calc_metric(data_dict, pred_dict)
            loss_dict.update(metric_dict)

            dataset = self.trainer.test_dataloaders[0].dataset
            json_dir_name = f"./jigsaw_eval/{dataset.category}_vol"
            if dataset.num_removal > 0:
                json_dir_name += f"_missing_{dataset.num_removal}"
            if dataset.num_redundancy > 0:
                json_dir_name += f"_redundant_{dataset.num_redundancy}"
            os.makedirs(json_dir_name, exist_ok=True)

            for b in range(B):
                gt_trans = data_dict["part_trans"][b][part_valids[b]]
                gt_rot = data_dict["part_rot"][b][part_valids[b]]._rot
                gt_rot = p3dt.matrix_to_quaternion(gt_rot)
                gt_trans_rots = (
                    torch.cat(
                        [
                            gt_trans,
                            gt_rot,
                        ],
                        dim=1,
                    )
                    .cpu()
                    .numpy()
                    .tolist()
                )
                pred_trans = pred_dict["trans"][b][part_valids_cpu[b]]
                pred_rot = pred_dict["rot"][b][part_valids_cpu[b]]
                pred_rot = p3dt.matrix_to_quaternion(torch.tensor(pred_rot)).numpy()
                pred_trans_rots = np.concatenate(
                    [
                        pred_trans,
                        pred_rot,
                    ],
                    axis=1,
                ).tolist()
                data = {
                    "name": data_dict["name"][b],
                    "num_parts": data_dict["num_parts"][b].item(),
                    "gt_trans_rots": gt_trans_rots,
                    "pred_trans_rots": pred_trans_rots,
                    "part_acc": raw_metric_dict["part_acc"][b].item(),
                    "rmse_t": raw_metric_dict["trans_rmse"][b].item(),
                    "rmse_r": raw_metric_dict["rot_rmse"][b].item(),
                    "shape_cd": raw_metric_dict["chamfer_distance"][b].item(),
                    "removal_pieces": data_dict["removal_pieces"][b],
                    "redundant_pieces": data_dict["redundant_pieces"][b],
                }

                json.dump(
                    data,
                    open(
                        f"{json_dir_name}/{data_dict['data_id'][b].item()}.json",
                        "w",
                    ),
                )

        return loss_dict

    def forward_pass(self, data_dict, mode, optimizer_idx):
        loss_dict = self.loss_function(
            data_dict, optimizer_idx=optimizer_idx, mode=mode
        )
        # in training we log for every step
        if mode == "train" and self.local_rank == 0:
            log_dict = {
                f"{mode}/{k}": v.item() if isinstance(v, torch.Tensor) else v
                for k, v in loss_dict.items()
            }
            data_name = [
                k
                for k in self.trainer.profiler.recorded_durations.keys()
                if "prepare_data" in k
            ][0]
            log_dict[f"{mode}/data_time"] = self.trainer.profiler.recorded_durations[
                data_name
            ][-1]
            self.log_dict(log_dict, logger=True, sync_dist=False, rank_zero_only=True)
        return loss_dict

    def configure_optimizers(self):
        """Build optimizer and lr scheduler."""
        lr = self.cfg.TRAIN.LR
        wd = self.cfg.TRAIN.WEIGHT_DECAY

        if wd > 0.0:
            params_dict = filter_wd_parameters(self)
            params_list = [
                {
                    "params": params_dict["no_decay"],
                    "weight_decay": 0.0,
                },
                {
                    "params": params_dict["decay"],
                    "weight_decay": wd,
                },
            ]
            optimizer = optim.AdamW(params_list, lr=lr)
        else:
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=0.0)

        if self.cfg.TRAIN.LR_SCHEDULER:
            assert self.cfg.TRAIN.LR_SCHEDULER.lower() in ["cosine"]
            total_epochs = self.cfg.TRAIN.NUM_EPOCHS
            warmup_epochs = int(total_epochs * self.cfg.TRAIN.WARMUP_RATIO)
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                total_epochs,
                max_lr=lr,
                min_lr=lr / self.cfg.TRAIN.LR_DECAY,
                warmup_steps=warmup_epochs,
            )
            return (
                [optimizer],
                [
                    {
                        "scheduler": scheduler,
                        "interval": "epoch",
                    }
                ],
            )
        return optimizer
