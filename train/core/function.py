# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os
from os import path as osp

import numpy as np
import torch
import torch.nn.functional as F

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back, transform_preds
from utils.vis import save_debug_images
import pprint
import sys
import json
import pickle
import mlflow

logger = logging.getLogger(__name__)



def train(config, train_loader, train_dataset, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # 에포크 손실 및 정확도 초기화
    epoch_loss_sum = 0
    epoch_acc_sum = 0
    num_batches = len(train_loader)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):

        # tsd = '/mnt/nas4/lyh/py_project/HRNet-for-Fashion-Landmark-Estimation.PyTorch/lyh_test/'


        # input_np = input.cpu().numpy()
        # target_np = target.cpu().numpy()
        # target_weight_np = target_weight.cpu().numpy()
        # print(f'shape of input : {input.shape}')
        # print(f'shape of target : {target.shape}')
        # print(f'shape of target_weight : {target_weight.shape}')
        # print(f'len of meta : {len(meta)}')
        # meta_json = meta


        # np.save(osp.join(tsd, 'input_np'), input_np)
        # np.save(osp.join(tsd, 'target_np'), target_np)
        # np.save(osp.join(tsd, 'target_weight_np'), target_weight_np)

        # with open(osp.join(tsd, 'meta.json'), 'w') as f:
        #     json.dump(meta, f, indent=4)

        # with open(osp.join(tsd, 'meta.pkl'), 'wb') as f:
        #     pickle.dump(meta, f)

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True).float()
        target_weight = target_weight.cuda(non_blocking=True).float()

        cat_ids = meta['category_id']
        c = meta['center'].numpy()
        s = meta['scale'].numpy()
        score = meta['score'].numpy()

        channel_mask = torch.zeros_like(target_weight).float()


        for j, cat_id in enumerate(cat_ids):
            rg = train_dataset.gt_class_keypoints_dict[int(cat_id)]
            index = torch.tensor([list(range(rg[0], rg[1]))], device=channel_mask.device, dtype=channel_mask.dtype).transpose(1,0).long()
            channel_mask[j].scatter_(0, index, 1)

        # compute output
        output = model(input)

        # print("type of output = model(input) : ", type(output))
        # b_output_np = output.detach().cpu().numpy()
        # np.save(osp.join(tsd, 'b_output_np'), b_output_np)

        if config.MODEL.TARGET_TYPE == 'gaussian':
            # block irrelevant channels in output
            output = output * channel_mask.unsqueeze(3)
            # print("after output shape : ", output.shape)
            preds_local, maxvals = get_final_preds(config, output.detach().cpu().numpy(), c, s)

            # print("type of output * channel_mask.unsqueeze(3) : ", type(output))
            # a_output_np = output.detach().cpu().numpy()
            # np.save(osp.join(tsd, 'a_output_np'), a_output_np)
            # print(f'type of preds_local : {type(preds_local)}')
            # print(f'type of maxvals : {type(maxvals)}')

            # np.save(osp.join(tsd, 'preds_local_np'), preds_local)
            # np.save(osp.join(tsd, 'maxvals_np'), maxvals)

            # print("preds_local shape : ", preds_local.shape)
            ########################################################
            # preds = preds_local.copy()
            # for tt in range(preds_local.shape[0]):
            #     preds[tt] = transform_preds(
            #         preds_local[tt], c[tt], s[tt],
            #         [config.MODEL.HEATMAP_SIZE[0], config.MODEL.HEATMAP_SIZE[1]]
            #     )

            # print(f'type of preds : {type(preds)}')
            # np.save(osp.join(tsd, 'preds_np'), preds)
            # preds = preds_local.copy()
            # for tt in range(preds_local.shape[0]):
            #     preds[tt] = transform_preds(
            #         preds_local[tt], c[tt], s[tt],
            #         [288, 384]
            #     )

            ########################################################

        elif config.MODEL.TARGET_TYPE == 'coordinate':
            heatmap, output = output
            
            # block irrelevant channels in output
            output = output * channel_mask
            preds_local, maxvals = get_final_preds(config, output.detach().cpu().numpy(), c, s, heatmap.detach().cpu().numpy())

            # Transform back from heatmap coordinate to image coordinate
            preds = preds_local.copy()
            for i in range(preds_local.shape[0]):
                preds[i] = transform_preds(
                    preds_local[i], c[i], s[i],
                    [config.MODEL.HEATMAP_SIZE[0], config.MODEL.HEATMAP_SIZE[1]]
                )
        else:
            raise NotImplementedError('{} is not implemented'.format(config.MODEL.TARGET_TYPE))

        loss = criterion(output, target, target_weight)

        print(f'loss value is : {loss}')
        #loss_np = loss.detach().cpu().numpy()
        #np.save(osp.join(tsd, 'loss_np'), loss_np)

        # sys.exit(0)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), input.size(0))
        epoch_loss_sum += loss.item()

        _, avg_acc, cnt, pred = accuracy("train", output.detach().cpu().numpy(), 
                                         target.detach().cpu().numpy(),
                                         train_dataset.target_type)

        acc.update(avg_acc, cnt)
        epoch_acc_sum += avg_acc

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if config.PRINT_FREQ == 1:
        # if i % config.PRINT_FREQ == 0:
        msg = 'Epoch: [{0}][{1}/{2}]\t' \
            'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
            'Speed {speed:.1f} samples/s\t' \
            'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
            'Loss {loss.val:.6f} ({loss.avg:.6f})\t' \
            'acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                speed=input.size(0)/batch_time.val,
                data_time=data_time, loss=losses, acc=acc)
        logger.info(msg)

        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', losses.val, global_steps)
        writer.add_scalar('train_acc', acc.val, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1

        prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)

        #################################################
        # print(meta.keys())
        # print(meta['image'])
        # pprint.pprint(meta['joints'])
        #print("shape : ", meta['joints'].shape)
        # #
        # print("preds")
        # pprint.pprint(preds)
        # print("shape : ", preds.shape)
        # #
        # print("preds_local")
        # pprint.pprint(preds_local)
        # print("shape : ", preds_local.shape)
        #
        #
        #sys.exit(0)
        ###############################################################

        save_debug_images(config, input, meta, target, preds_local, output, prefix)
        # save_debug_images(config, input, meta, target, preds, output, prefix)

        # mlflow에 손실과 정확도 기록
        mlflow.log_metric("train_loss", losses.val, step=epoch * len(train_loader) + i)
        mlflow.log_metric("train_acc", acc.val, step=epoch * len(train_loader) + i)

        # 배치 손실 누적
        epoch_loss_sum += loss.item()

        
#
def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None, epoch=None):
    if epoch is None:
        raise ValueError("Epoch value must be provided to the validate function.")

    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # 에포크 손실 및 정확도 초기화
    epoch_loss_sum = 0
    epoch_acc_sum = 0
    num_batches = len(val_loader)

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 7))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            
            target = target.cuda(non_blocking=True).float()
            target_weight = target_weight.cuda(non_blocking=True).float()

            cat_ids = meta['category_id']
            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            channel_mask = torch.zeros_like(target_weight).float()
            for j, cat_id in enumerate(cat_ids):
                rg = val_dataset.gt_class_keypoints_dict[int(cat_id)]
                index = torch.tensor([list(range(rg[0], rg[1]))], device=channel_mask.device, dtype=channel_mask.dtype).transpose(1,0).long()
                channel_mask[j].scatter_(0, index, 1)
                
            # compute output
            output = model(input)

            if config.MODEL.TARGET_TYPE == 'gaussian':
                if config.TEST.FLIP_TEST:
                    # this part is ugly, because pytorch has not supported negative index
                    # input_flipped = model(input[:, :, :, ::-1])
                    input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                    input_flipped = torch.from_numpy(input_flipped).cuda()
                    outputs_flipped = model(input_flipped)

                    if isinstance(outputs_flipped, list):
                        output_flipped = outputs_flipped[-1]
                    else:
                        output_flipped = outputs_flipped
                    output_flipped = output_flipped.cpu().numpy()
                    
                    category_id_list = meta['category_id'].cpu().numpy().copy()
                    for j, category_id in enumerate(category_id_list):
                        output_flipped[j, :, :, :] = flip_back(output_flipped[j, None],
                                                val_dataset.flip_pairs[category_id-1],
                                                config.MODEL.HEATMAP_SIZE[0])
                    output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                    # feature is not aligned, shift flipped heatmap for higher accuracy
                    if config.TEST.SHIFT_HEATMAP:
                        output_flipped[:, :, :, 1:] = \
                            output_flipped.clone()[:, :, :, 0:-1]

                    output = (output + output_flipped) * 0.5

                # block irrelevant channels in output
                output = output * channel_mask.unsqueeze(3)
                preds_local, maxvals = get_final_preds(config, output.detach().cpu().numpy(), c, s)

                # Transform back from heatmap coordinate to image coordinate
                preds = preds_local.copy()
                for tt in range(preds_local.shape[0]):
                    preds[tt] = transform_preds(
                        preds_local[tt], c[tt], s[tt],
                        [config.MODEL.HEATMAP_SIZE[0], config.MODEL.HEATMAP_SIZE[1]]
                    )
                
            else:
                raise NotImplementedError('{} is not implemented'.format(config.MODEL.TARGET_TYPE))
            
            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            
            _, avg_acc, cnt, pred = accuracy("valid",output.detach().cpu().numpy(), 
                                        target.detach().cpu().numpy(),
                                        val_dataset.target_type)
            acc.update(avg_acc, cnt)

            # mlflow에 손실과 정확도 기록
            mlflow.log_metric("val_loss", losses.val, step=i)
            mlflow.log_metric("val_acc", acc.val, step=i)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals

            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            all_boxes[idx:idx + num_images, 6] = meta['category_id'].cpu().numpy().astype(int)

            image_path.extend(meta['image'])

            idx += num_images

            msg = 'Test: [{0}/{1}]\t' \
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                    'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                    'acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time,
                        loss=losses, acc=acc)
            logger.info(msg)

            prefix = '{}_{}'.format(
                os.path.join(output_dir, 'val'), i
            )

            print("save validation")
            save_debug_images(config, input, meta, target, preds_local, output, prefix)

            # 손실 계산
            loss = criterion(output, target, target_weight)

            # 배치 손실 누적
            epoch_loss_sum += loss.item()

            # 정확도 계산
            _, avg_acc, cnt, pred = accuracy("valid", output.detach().cpu().numpy(), 
                                             target.detach().cpu().numpy(),
                                             val_dataset.target_type)

            # 배치 정확도 누적
            epoch_acc_sum += avg_acc

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )


        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_AP',
                perf_indicator,
                global_steps
            )
            
            writer_dict['valid_global_steps'] = global_steps + 1

    # 에포크가 끝난 후 평균 손실 및 정확도 계산 및 출력
    epoch_loss_avg = epoch_loss_sum / num_batches
    epoch_acc_avg = epoch_acc_sum / num_batches
    print(f'Epoch [{epoch+1}] Validation Average Loss: {epoch_loss_avg:.6f}, Average Accuracy: {epoch_acc_avg:.3f}')

    # 에포크가 끝난 후 손실과 정확도 기록
    mlflow.log_metric("epoch_val_loss", epoch_loss_sum / num_batches, step=epoch)
    mlflow.log_metric("epoch_val_acc", epoch_acc_sum / num_batches, step=epoch)

    return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.4f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
