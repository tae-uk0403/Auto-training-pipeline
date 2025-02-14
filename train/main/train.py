# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import mlflow
import mlflow.pytorch

import _init_paths
from config import cfg
from config import update_config

from core.loss import JointsMSELoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

import dataset
import models
import sys
import numpy as np


#
def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")
    # general
    parser.add_argument(
        "--cfg", help="experiment configure file name", required=True, type=str
    )

    parser.add_argument(
        "--run_name",  # run_name 추가
        help="Name of the MLflow run",
        type=str,
        default="default_run",
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    # philly
    parser.add_argument("--modelDir", help="model directory", type=str, default="")
    parser.add_argument("--logDir", help="log directory", type=str, default="")
    parser.add_argument("--dataDir", help="data directory", type=str, default="")
    parser.add_argument(
        "--prevModelDir", help="prev Model directory", type=str, default=""
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, "train")

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    if torch.cuda.is_available():
        train_batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU * torch.cuda.device_count()
        test_batch_size = cfg.TEST.BATCH_SIZE_PER_GPU * torch.cuda.device_count()
        logger.info("Let's use %d GPUs!" % torch.cuda.device_count())

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval("models." + cfg.MODEL.NAME + ".get_pose_net")(
        cfg, is_train=True
    ).cuda()

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, "../models", cfg.MODEL.NAME + ".py"), final_output_dir
    )
    # logger.info(pprint.pformat(model))

    writer_dict = {
        "writer": SummaryWriter(log_dir=tb_log_dir),
        "train_global_steps": 0,
        "valid_global_steps": 0,
    }

    # dump_input = torch.rand((1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])).cuda()
    # writer_dict['writer'].add_graph(model, (dump_input, ))
    # logger.info(get_model_summary(model, dump_input))

    model = torch.nn.DataParallel(model)
    # model = torch.nn.DataParallel(model, device_ids=cfg.GPUS)

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        cfg=cfg,
        target_type=cfg.MODEL.TARGET_TYPE,
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT,
    ).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_dataset = eval("dataset." + cfg.DATASET.DATASET)(
        cfg,
        cfg.DATASET.ROOT,
        cfg.DATASET.TRAIN_SET,
        True,
        transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    valid_dataset = eval("dataset." + cfg.DATASET.DATASET)(
        cfg,
        cfg.DATASET.ROOT,
        cfg.DATASET.TEST_SET,
        False,
        transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    """ Due to imbalance of dataset, adjust sampling weight for each class
        according to class distribution
    """
    cls_prop = train_dataset.cls_stat / train_dataset.cls_stat.sum()
    cls_weights = 1 / (cls_prop + 0.02)
    str_index = "Class idx  "
    str_prop = "Proportion "
    str_weigh = "Weights    "
    for i in range(len(cls_prop)):
        str_index += "| %5d " % (i)
        str_prop += "| %5.2f " % cls_prop[i]
        str_weigh += "| %5.2f " % cls_weights[i]
    logger.info("Training Data Analysis:")
    logger.info(str_index)
    logger.info(str_prop)
    logger.info(str_weigh)
    sample_list_of_cls = train_dataset.sample_list_of_cls
    sample_list_of_weights = list(map(lambda x: cls_weights[x], sample_list_of_cls))
    train_sampler = torch.utils.data.WeightedRandomSampler(
        sample_list_of_weights, len(train_dataset)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        # batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        batch_size=train_batch_size,
        # shuffle=cfg.TRAIN.SHUFFLE,
        sampler=train_sampler,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        # batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    best_perf = 0.0
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(final_output_dir, "checkpoint.pth")

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint["epoch"]
        best_perf = checkpoint["perf"]
        last_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info(
            "=> loaded checkpoint '{}' (epoch {})".format(
                checkpoint_file, checkpoint["epoch"]
            )
        )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR, last_epoch=last_epoch
    )
    logger.info("=> Start training...")

    # MLflow experiment 설정
    mlflow.set_experiment("keypoint_detection_whole_body")

    with mlflow.start_run(run_name=args.run_name):
        # 하이퍼파라미터 기록
        mlflow.log_param("learning_rate", cfg.TRAIN.LR)
        mlflow.log_param("batch_size", train_batch_size)
        # 필요한 다른 파라미터도 기록 가능

        for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
            lr_scheduler.step()

            # train for one epoch
            train(
                cfg,
                train_loader,
                train_dataset,
                model,
                criterion,
                optimizer,
                epoch,
                final_output_dir,
                tb_log_dir,
                writer_dict,
            )

            # evaluate on validation set
            perf_indicator = validate(
                cfg,
                valid_loader,
                valid_dataset,
                model,
                criterion,
                final_output_dir,
                tb_log_dir,
                writer_dict,
                epoch,  # epoch 인자 전달
            )

            # 성능 지표 기록
            mlflow.log_metric("performance", perf_indicator, step=epoch)

            if perf_indicator >= best_perf:
                best_perf = perf_indicator
                best_model = True
            else:
                best_model = False

            # 모델 저장
            if best_model:
                mlflow.pytorch.log_model(model, "best_model")

            logger.info("=> saving checkpoint to {}".format(final_output_dir))
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model": cfg.MODEL.NAME,
                    "state_dict": model.state_dict(),
                    "best_state_dict": model.module.state_dict(),
                    "perf": perf_indicator,
                    "optimizer": optimizer.state_dict(),
                },
                best_model,
                final_output_dir,
                f"epoch-{epoch}.pth",
            )
            logger.info("# Best AP {}".format(best_perf))

        # 최종 모델 상태 저장
        final_model_state_file = os.path.join(final_output_dir, "final_state.pth")
        logger.info("=> saving final model state to {}".format(final_model_state_file))
        torch.save(model.module.state_dict(), final_model_state_file)
        writer_dict["writer"].close()

        # 최종 모델을 MLflow에 저장
        mlflow.pytorch.log_model(model, "final_model")


if __name__ == "__main__":
    main()
