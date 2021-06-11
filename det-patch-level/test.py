from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import warnings

warnings.filterwarnings("ignore")


def det_evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size, images_folder, name="init",
             step=1090000, fp_flag=False, NEG_SAMPLE_INTERVAL=1, interval_sign=' '):
    model.eval()
    PATCH_SAVE_TXT = 'test_txts/wei_%s_step%d'%(name, int(step)) + '_' + str(img_size[0]) +'nms'+str(nms_thres)+'conf'+str(conf_thres)+'.txt'
    # PATCH_SAVE_TXT = 'Z:/wei/All_Of_Them/50Slides/wei_%s_step%d' % (name, int(step)) + '_' + str(
    #     img_size[0]) + 'nms' + str(nms_thres) + 'conf' + str(conf_thres) + '.txt'
    # PATCH_SAVE_TXT = 'Z:/wei/PCDD/test_dataset/wei_%s_step%d'%(name, int(step)) + '_' + str(img_size[0]) +'nms'+str(nms_thres)+'conf'+str(conf_thres)+'.txt'
    save_str = ''
    # Get dataloader
    dataset = ListDataset(path, folder=images_folder, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels, mloss = [], []
    fp_nums = []
    neg_sample_times = 0
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (img_fns, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        new_img_size = img_size[0]
        if targets is None:
            print("Opps.. There're no targets be found")
            exit()
        eval_flag = True
        if targets.shape[0] == 0:
            eval_flag = neg_sample_times % NEG_SAMPLE_INTERVAL == 0
            neg_sample_times += 1
        if not eval_flag:
            continue
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2] *= img_size[0]
        targets[:, 3] *= img_size[1]
        targets[:, 4] *= img_size[0]
        targets[:, 5] *= img_size[1]

        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        targets = Variable(targets, requires_grad=False).to(device)
        with torch.no_grad():
            loss, outputs = model(imgs, targets=targets, fp_flag=fp_flag)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
            mloss.append(loss)
        sample_metric = get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
        sample_metrics += sample_metric 
        # po = [o for o in outputs if o is not None]
    #     po = [o for o in outputs]
    #     # for _o, sample, img_fn in zip(po, sample_metric, img_fns):
    #     for _o, img_fn in zip(po, img_fns):
    #         # true_positives, _, _ = sample
    #         # fp_nums.append(len(np.where(true_positives == 0)[0]))
    #         if _o is None: continue
    #         # gx, gy = img_fn.split('/')[-1][:-4].split('_')[-2], img_fn.split('/')[-1][:-4].split('_')[-1]
    #         # gx, gy = int(gx), int(gy)
    #
    #         dataId = img_fn.split('/')[-3]
    #         slide_name = img_fn.split('/')[-2]
    #         dataId = dataId + '_' + slide_name
    #
    #         # old_wh = int(img_fn[:-4].split('_')[-1])
    #         # old_wh = w
    #         old_wh = 1024
    #         # old_wh = new_img_size
    #         global_outputs = _o.clone()
    #         global_outputs[:, 0] = ((global_outputs[:, 0]) / new_img_size) * old_wh
    #         global_outputs[:, 1] = ((global_outputs[:, 1]) / new_img_size) * old_wh
    #         global_outputs[:, 2] = ((global_outputs[:, 2]) / new_img_size) * old_wh
    #         global_outputs[:, 3] = ((global_outputs[:, 3]) / new_img_size) * old_wh
    #         xmins = global_outputs[:, 0].tolist()
    #         ymins = global_outputs[:, 1].tolist()
    #         xmaxs = global_outputs[:, 2].tolist()
    #         ymaxs = global_outputs[:, 3].tolist()
    #         scores = global_outputs[:, 4].tolist()
    #
    #         id = img_fn.split('/')[-1][:-4]
    #         # id = dataId
    #         for score, xmin, ymin, xmax, ymax in zip(scores, xmins, ymins, xmaxs, ymaxs):
    #             save_str += '%s%s%f%s%f%s%f%s%f%s%f\n' % (
    #             id, interval_sign, score, interval_sign, xmin, interval_sign, ymin, interval_sign, xmax, interval_sign,
    #             ymax)
    #
    # with open(PATCH_SAVE_TXT, 'w') as f:
    #     f.write(save_str)
    # exit()
    # Concatenate sample statistics
    if sample_metrics != []:
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    else:
        precision, recall, AP, f1, ap_class = torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), [0]

    return precision, recall, AP, f1, ap_class, fp_nums, torch.cat(mloss).mean()

def cls_evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size, images_folder, name="init",
             step=1090000, fp_flag=False, NEG_SAMPLE_INTERVAL=1, interval_sign=' '):
    model.eval()
    # Get dataloader
    dataset = ListDataset(path, folder=images_folder, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mloss = []
    neg_sample_times = 0
    precisions, recalls, ious, dices = [], [], [], []  # List of tuples (TP, confs, pred)
    for batch_i, (img_fns, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        if targets is None:
            print("Opps.. There're no targets be found")
            exit()
        eval_flag = True
        if targets.shape[0] == 0:
            eval_flag = neg_sample_times % NEG_SAMPLE_INTERVAL == 0
            neg_sample_times += 1
        if not eval_flag:
            continue

        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        targets = Variable(targets, requires_grad=False).to(device)
        with torch.no_grad():
            loss, outputs = model(imgs, targets=targets, fp_flag=fp_flag)
            print('Val loss:', loss.item())
            mloss.append(loss.item())
        pred_mask = outputs[..., 4]
        target_mask = outputs[..., 5] * (1/0.95)
        # print('pred_mask', pred_mask.shape, pred_mask.unique())
        # print('target_mask', target_mask.shape, target_mask.unique())
        out, tar = pred_mask.data.cpu().numpy(), target_mask.data.cpu().numpy()
        dice = cal_dice(out, tar)
        iou = iou_score(out, tar)
        precision, recall = cal_prerecall(out, tar)
        # print(dice, iou)
        # exit()
        precisions.append(precision)
        recalls.append(recall)
        ious.append(iou)
        dices.append(dice)

    precisions, recalls, ious, dices = np.concatenate(precisions), np.concatenate(recalls), np.concatenate(ious), np.concatenate(dices)
    ap_class, fp_nums = [], []
    return precisions, recalls, ious, dices, ap_class, fp_nums, np.array(mloss).mean()

def cal_prerecall(output, target):
    smooth = 1e-5
    tp = (output * target).sum(axis=1)
    fp = output.sum(axis=1) - tp
    fn = target.sum(axis=1) - tp
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return precision, recall

def cal_dice(output, target):
    smooth = 1e-5
    intersection = (output * target).sum(axis=1)
    return (2. * intersection + smooth) / \
        (output.sum(axis=1) + target.sum(axis=1) + smooth)

def iou_score(output, target):
    smooth = 1e-5
    output_ = output > 0.7
    target_ = target > 0.7
    intersection = (output_ & target_).sum(axis=1)
    union = (output_ | target_).sum(axis=1)

    return (intersection + smooth) / (union + smooth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3-sRMB-v02.cfg",
                        help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="sRMBv02_pcdd1024/yolov3_ckpt_step_869998.pth",
                        help="path to weights file")
    parser.add_argument("--class_path", type=str, default="Z:/wei/PyTorch-YOLOv3-master/data/custom/classes.names",
                        help="path to class label file")
    parser.add_argument("--folder", type=str, default="train_images1024_new")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.3, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=tuple, default=(1024, 1024), help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    evaluate = det_evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data_config = parse_data_config(opt.data_config)
    # valid_path = data_config["valid"]
    valid_path = 'E:/wei/PCDD_all/data_sets/images1024_valid.txt'
    class_names = load_classes('data/custom/classes.names')

    # Initiate model
    model = Darknet(opt.model_def, img_size=opt.img_size, use_final_loss=True).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class, fp_nums = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
        fp_flag=True,
        images_folder=opt.folder,
        name="sRMBv02_pcdd1248_pos",
        step=870000,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
