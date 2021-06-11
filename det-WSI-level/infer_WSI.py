from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
# from apex import amp

import os
import cv2
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torchvision.models as models
from torch.autograd import Variable
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
Image.MAX_IMAGE_PIXELS = 500000000
_ROOT_ = 'D:/WSI_analysis/det'
_ROOT0_ = 'D:/WSI_analysis'

def distanceConserve(x, y, xlist, ylist, thres=1000):
    for xi, yi in zip(xlist, ylist):
        if ((x - xi) ** 2 + (y - yi) ** 2) ** 0.5 <= thres:
            return False
    return True


def buildMinsMaxsWithThres(maxargsort, minargsort, one_term_len, xs, ys, disThres):
    maxs, mins = [], []
    i = 0
    while len(maxs) < one_term_len:
        x, y = xs[maxargsort[i]], ys[maxargsort[i]]
        if len(maxs) > 1:
            xlist, ylist = xs[torch.stack(maxs)].tolist(), ys[torch.stack(maxs)].tolist()
        else:
            xlist, ylist = xs[maxs].tolist(), ys[maxs].tolist()
            if type(xlist) is float or type(ylist) is float:
                xlist, ylist = [xlist], [ylist]
        if distanceConserve(x, y, xlist, ylist, thres=disThres):
            maxs += [maxargsort[i]]
        i += 1
    maxs = torch.stack(maxs)
    i = 0
    while len(mins) < one_term_len:
        x, y = xs[minargsort[i]], ys[minargsort[i]]
        if len(mins) > 1:
            xlist, ylist = xs[torch.stack(mins)].tolist(), ys[torch.stack(mins)].tolist()
        else:
            xlist, ylist = xs[mins].tolist(), ys[mins].tolist()
            if type(xlist) is float or type(ylist) is float:
                xlist, ylist = [xlist], [ylist]
        if distanceConserve(x, y, xlist, ylist, thres=disThres):
            mins += [minargsort[i]]
        i += 1
    mins = torch.stack(mins)
    return mins, maxs


def getMaxsMinsFromYolo(yolo_map, one_term_len, disThres, xs, ys):
    maxargsort = torch.argsort(yolo_map.reshape(-1))
    minargsort = torch.argsort(-1.0 * yolo_map.reshape(-1))
    if disThres == 0:
        maxs = maxargsort[:one_term_len]
        mins = minargsort[:one_term_len]
    else:
        mins, maxs = buildMinsMaxsWithThres(maxargsort, minargsort, one_term_len, xs, ys, disThres)
    return mins, maxs


def prepareSequence(xs, ys, fm, mins, maxs):
    data = torch.zeros(len(mins)+len(maxs), fm.shape[0])
    data_index = 0
    for xi, yi in zip(xs[mins], ys[mins]):
        data[data_index] = fm[:, int(xi.item()), int(yi.item())]
        data_index += 1
    for xi, yi in zip(xs[maxs], ys[maxs]):
        data[data_index] = fm[:, int(xi.item()), int(yi.item())]
        data_index += 1
    return data


def get_sequence(yolos, fms, anchor_num=2, model_dim=7, disThres=0):
    _mins = []
    _maxs = []
    outputs = []
    if not isMNV2:
        xs = torch.zeros(anchor_num, fms.shape[-2], fms.shape[-1])
        ys = torch.zeros(anchor_num, fms.shape[-2], fms.shape[-1])
        for i in range(fms.shape[-1]):
            ys[:, :, i] = i
        for i in range(fms.shape[-2]):
            xs[:, i, :] = i
        xs = xs.reshape(-1)
        ys = ys.reshape(-1)
        for yolo in yolos:
            if yolo.shape[-1] != fms.shape[-1]:
                yolo = torch.max_pool2d(yolo, int(yolo.shape[-1]/fms.shape[-1]), int(yolo.shape[-1]/fms.shape[-1]))
            yolo = yolo.view(anchor_num, model_dim, yolo.shape[1], yolo.shape[2]).permute(0, 2, 3,
                                                                                                           1).contiguous()
            outputs.append(yolo)
            conf = torch.sigmoid(yolo[..., 4] * torch.exp(-1 * yolo[..., 5]))  # conf
            fp_conf = torch.sigmoid(yolo[..., 5])  # fp_conf
            mins, maxs = getMaxsMinsFromYolo(conf, 50, disThres=disThres, xs=xs, ys=ys)
            fpmins, fpmaxs = getMaxsMinsFromYolo(fp_conf, 50, disThres=disThres, xs=xs, ys=ys)
            _mins = _mins + [mins] + [fpmins]
            _maxs = _maxs + [fpmaxs] + [maxs]
            # _mins = _mins + [mins]
            # _maxs = _maxs + [maxs]
        _mins = torch.cat(_mins)
        _maxs = torch.cat(_maxs)
    else:
        xs = torch.zeros(fms.shape[-2], fms.shape[-1])
        ys = torch.zeros(fms.shape[-2], fms.shape[-1])
        for i in range(fms.shape[-1]):
            ys[:, i] = i
        for i in range(fms.shape[-2]):
            xs[i, :] = i
        xs = xs.reshape(-1)
        ys = ys.reshape(-1)
        for yolo in yolos:
            if yolo.shape[-1] != fms.shape[-1]:
                yolo = torch.max_pool2d(yolo, int(yolo.shape[-1]/fms.shape[-1]), int(yolo.shape[-1]/fms.shape[-1]))
            outputs.append(yolo)
            conf = torch.sigmoid(yolo[0, ...])  # conf
            mins, maxs = getMaxsMinsFromYolo(conf, 100, disThres=disThres, xs=xs, ys=ys)
            _mins = _mins + [mins]
            _maxs = _maxs + [maxs]
        _mins = torch.cat(_mins)
        _maxs = torch.cat(_maxs)
    return prepareSequence(xs, ys, fms, _mins, _maxs), prepareSequence(ys, xs, fms, _mins, _maxs), _mins, _maxs, torch.stack(outputs)


def name2slidename(name):
    slide_name = ''
    for s in name.split('-')[:-2]:
        slide_name += s + '-'
    return slide_name[:-1]


def prepare_xys(dataset):
    xys = {}
    names = {}
    out_xys = {}
    for info in dataset.imgs_info:
        name = info['name']
        if len(name.split('--')) == 1:
            x, y = name.split('-')[-2:]
        elif len(name.split('--')) == 2:
            if '-' in name.split('--')[0] and '-' not in name.split('--')[1]:
                x = name.split('--')[0].split('-')[-1]
                y = '-' + name.split('--')[1]
            elif '-' in name.split('--')[0] and '-' in name.split('--')[1]:
                x, y = name.split('--')[1].split('-')
                x = '-' + x
            else:
                x, y = name.split('--')[1].split('-')
        elif len(name.split('--')) == 3:
            x, y = name.split('--')[-2:]
            x, y = '-' + x, '-' + y
        slide_name = name2slidename(name)
        if slide_name not in xys: xys[slide_name] = []
        xys[slide_name].append([int(x), int(y)])
        if slide_name not in names: names[slide_name] = []
        names[slide_name].append(name)
    for slide_name in xys:
        xs = np.unique(np.array(xys[slide_name])[:, 0])
        ys = np.unique(np.array(xys[slide_name])[:, 1])
        x2ind, y2ind = {}, {}
        for ind, xi in enumerate(xs):
            x2ind[xi] = ind
        for ind, yi in enumerate(ys):
            y2ind[yi] = ind
        for name, xy in zip(names[slide_name], xys[slide_name]):
            out_xys[name] = [x2ind[xy[0]], y2ind[xy[1]]]

    return out_xys


def get_overlapmask(yolo, oldyolo, anchor_num, model_dim):
    yolo = yolo.view(anchor_num, model_dim, yolo.shape[1], yolo.shape[2]).permute(0, 2, 3,
                                                                                  1).contiguous()
    oldyolo = oldyolo.view(anchor_num, model_dim, yolo.shape[1], yolo.shape[2]).permute(0, 2, 3,
                                                                                  1).contiguous()
    conf = yolo[..., 4] * torch.exp(-1 * yolo[..., 5])  # conf
    fp_conf = yolo[..., 5]  # fp_conf
    oldconf = oldyolo[..., 4] * torch.exp(-1 * oldyolo[..., 5])  # conf
    oldfp_conf = oldyolo[..., 5]  # fp_conf
    mask = conf > oldconf
    fp_mask = fp_conf > oldfp_conf
    for i in range(1, anchor_num):
        mask[0] |= mask[i]
        fp_mask[0] |= fp_mask[i]
    return mask[0] | fp_mask[0]


def determinating_path(path):
    if os.name == 'posix':
        path = path.replace('Z:', '/mnt/194_z')
    return path


def evaluate(model, list_path, iou_thres, conf_thres, nms_thres, img_size, batch_size, images_folder, labels_folder='labels', SAVE_TXT='pcdd_val.txt', fp_flag=False, interval_sign=',', overlap = 100, level=0, center_side=40000, name='val'):
    
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    # Tensor = torch.FloatTensor
    font = cv2.FONT_HERSHEY_SIMPLEX
    model.eval()
    ######################################
    seq_root = r'%s_output\sfyall_sequencesmnv2_forvis' % _ROOT_
    # seq_root = r'%s_output\xyw_icnmanualchecked' % _ROOT_
    # seq_root = r'%s_output\sfy1&2_sequencesicn_clsNewXYconverse' % _ROOT_)
    seq_root = determinating_path(seq_root.replace('\\', '/'))
    os.makedirs(seq_root, exist_ok=True)
    os.makedirs(seq_root+'disconserve0', exist_ok=True)
    ######################################
    dataset = SrpDataset(
        list_path, 
        convert_channel=False, 
        img_size=(img_size, img_size), 
        sample_wh=(img_size * (level+1), img_size * (level+1)),
        overlap=overlap,
        center_side=center_side,
        augment=False, 
        multiscale=False, 
        normalized_labels=True, 
        fp_dict={},
        collect_fn=SrpDataset.sliding_window,
        saved_root=seq_root + 'disconserve0'
        )
    if isMNV2:
        classifier = nn.Conv2d(1280, 2, 1).to(device)
        classifier.eval()
        with torch.no_grad():
            for clsp, p in zip(classifier.parameters(), model.classifier[1].parameters()):
                if len(clsp.shape) == 4: clsp.copy_(p.reshape(p.shape[0], p.shape[1], 1, 1))
                else: clsp.copy_(p)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn
    )

    if 'icn' in seq_root:
        anchor_num, model_dim = 3, 7
    if 'yolov3' in seq_root:
        anchor_num, model_dim = 2, 7
    if 'tiny' in seq_root:
        anchor_num, model_dim = 3, 6
    if 'yolov4' in seq_root:
        anchor_num, model_dim = 2, 7
    if 'mnv2' in seq_root:
        anchor_num, model_dim = 1, 1

    fm_side0 = img_size / 16
    fm_side1 = img_size / 32
    times = center_side / img_size
    fms = {}
    yolos = {}
    fulfill_flag = {}
    xys = prepare_xys(dataset)
    allend0 = int([i for i in range(0, center_side, img_size-overlap)][-1] / 16) + int(img_size / 16)
    allend1 = int([i for i in range(0, center_side, img_size-overlap)][-1] / 32) + int(img_size / 32)
    for batch_i, (srp_files, names, sample_pts, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        print('Data loaded!')
        if os.path.exists(os.path.join(seq_root + 'disconserve0', name2slidename(names[0]) + '.pth')):
            print('skip', os.path.join(seq_root + 'disconserve0', name2slidename(names[0]) + '.pth'))
            continue
        if imgs is None: continue
        start = datetime.datetime.now()
        if isMNV2:
            with torch.no_grad():
                featuremap = model.features(imgs.to(device))
                featuremap = nn.functional.adaptive_avg_pool2d(featuremap, int(featuremap.shape[-1]/10))
                cls_map = classifier(featuremap)
            output = {'yolo': [cls_map[:, 0:1, :, :].cpu().clone()], 'feature_map': [featuremap.cpu().clone()]}
        else:
            with torch.no_grad():
                output = model(imgs.to(device), fp_flag=fp_flag, x_pre_filters=str(int(anchor_num*model_dim)))
        print('Data computed!')
        for i, name in enumerate(names):
            yolo, fm = [y[i] for y in output['yolo']], [f[i] for f in output['feature_map']]
            allends = [int([i for i in range(0, center_side, img_size-overlap)][-1] / (img_size / y.shape[-1])) + int(img_size / (img_size / y.shape[-1])) for y in yolo]
            fmsides = [img_size / (img_size / y.shape[-1]) for y in yolo]
            slide_name = name2slidename(name)
            xy = xys[name]
            if slide_name not in yolos:
                yolos[slide_name] = [
                    torch.zeros(int(anchor_num * model_dim), allend, allend)
                    for allend in allends
                ]
            print('yolo allocated')
            if slide_name not in fms:
                fms[slide_name] = [
                    torch.zeros(fm[_i].shape[0], allend, allend)
                    for _i, allend in enumerate(allends)
                ]
            print('fm allocated')
            for _i, (allend, fm_side) in enumerate(zip(allends, fmsides)):
                xoffset = 0 if xy[0] == 0 else int(overlap / (img_size / yolo[_i].shape[-1]))
                yoffset = 0 if xy[1] == 0 else int(overlap / (img_size / yolo[_i].shape[-1]))
                xstart = int(xy[0] * (fm_side - xoffset))
                ystart = int(xy[1] * (fm_side - yoffset))
                xend = int(xstart + fm_side)
                yend = int(ystart + fm_side)
                if xoffset != 0:
                    xoverlap_mask = get_overlapmask(yolo[_i][:, :xoffset, :], yolos[slide_name][_i][:, xstart:xstart+xoffset, ystart:yend])
                    yoloxmask = torch.cat([xoverlap_mask.unsqueeze(0) for _ in range(yolo[_i].shape[0])])
                    yolos[slide_name][_i][:, xstart:xstart + xoffset, ystart:yend][yoloxmask] = yolo[_i][:, :xoffset, :][yoloxmask]
                if yoffset != 0:
                    yoverlap_mask = get_overlapmask(yolo[_i][:, :, :yoffset], yolos[slide_name][_i][:, xstart:xend, ystart:ystart+yoffset])
                    yoloymask = torch.cat([yoverlap_mask.unsqueeze(0) for _ in range(yolo[_i].shape[0])])
                    yolos[slide_name][_i][:, xstart:xend, ystart:ystart + yoffset][yoloymask] = yolo[_i][:, :, :yoffset][yoloymask]
                yolos[slide_name][_i][:, xstart + xoffset:xend, ystart:yend] = yolo[_i][:, xoffset:, :]
                yolos[slide_name][_i][:, xstart:xend, ystart+yoffset:yend] = yolo[_i][:, :, yoffset:]
                if xoffset != 0:
                    fmxmask = torch.cat([xoverlap_mask.unsqueeze(0) for _ in range(fm[_i].shape[0])])
                    fms[slide_name][_i][:, xstart:xstart+xoffset, ystart:yend][fmxmask] = fm[_i][:, :xoffset, :][fmxmask]
                if yoffset != 0:
                    fmymask = torch.cat([yoverlap_mask.unsqueeze(0) for _ in range(fm[_i].shape[0])])
                    fms[slide_name][_i][:, xstart:xend, ystart:ystart+yoffset][fmymask] = fm[_i][:, :, :yoffset][fmymask]
                fms[slide_name][_i][:, xstart+xoffset:xend, ystart:yend] = fm[_i][:, xoffset:, :]
                fms[slide_name][_i][:, xstart:xend, ystart+yoffset:yend] = fm[_i][:, :, yoffset:]

            del xys[name]
            fulfill_flag[slide_name] = True
            for n in xys:
                if slide_name == name2slidename(n):
                    fulfill_flag[slide_name] = False
            print('\n', xy, name, slide_name, fulfill_flag[slide_name], '\n')
            if fulfill_flag[slide_name]:
                for _i in range(len(fms[slide_name])):
                    print('fms[slide_name][%s]'%_i, fms[slide_name][_i].shape)
                    print('yolos[slide_name][%s]'%_i, yolos[slide_name][_i].shape)
                    if fms[slide_name][_i].shape[-1] != allends[0]:
                        fms[slide_name][_i] = F.interpolate(fms[slide_name][_i].unsqueeze(0), size=allends[0], mode='nearest')[0]
                        print('Changed: fms[slide_name][%s]'%_i, fms[slide_name][_i].shape)
                fms[slide_name] = torch.cat(fms[slide_name])
                print('fms[slide_name]', fms[slide_name].shape)
                seq_dataxy, seq_datayx, _mins, _maxs, outputs = get_sequence(yolos[slide_name], fms[slide_name], disThres=0, anchor_num=anchor_num, model_dim=model_dim)
                expseq_dis0 = {'dataxy': seq_dataxy, 'datayx': seq_datayx, 'mins': _mins, 'maxs': _maxs, 'outputs': outputs}
                torch.save(expseq_dis0, os.path.join(seq_root+'disconserve0', slide_name + '.pth'))
                seq_dataxy, seq_datayx, _mins, _maxs, outputs = get_sequence(yolos[slide_name], fms[slide_name], disThres=10, anchor_num=anchor_num, model_dim=model_dim)
                expseq = {'dataxy': seq_dataxy, 'datayx': seq_datayx, 'mins': _mins, 'maxs': _maxs, 'outputs': outputs}
                torch.save(expseq, os.path.join(seq_root, slide_name + '.pth'))
                del yolos[slide_name], fms[slide_name]
        print(datetime.datetime.now()-start)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="%s/config/yolov3-sRMB-v02.cfg" % _ROOT_, help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="%s/sRMBv02/yolov3_custom_best.pth" % _ROOT_, help="path to weights file")
    # parser.add_argument("--weights_path", type=str, default="%s/sRMBv02-cls/yolov3_ckpt_step_1649998.pth" % _ROOT_, help="if specified starts from checkpoint model")
    # parser.add_argument("--model_def", type=str, default="%s/yolo-tiny/yolov3-tiny.cfg" % _ROOT_, help="path to model definition file")
    # parser.add_argument("--weights_path", type=str, default="%s/yolo-tiny/yolo-tiny-epoch80.pth" % _ROOT_, help="path to weights file")
    # parser.add_argument("--model_def", type=str, default="%s/config/yolov3-custom.cfg" % _ROOT_, help="path to model definition file")
    # parser.add_argument("--weights_path", type=str, default="%s/darknet53_pcdd1024/yolov3_ckpt_step_349998-the_best_one.pth" % _ROOT_, help="path to weights file")
    # parser.add_argument("--model_def", type=str, default="%s/config/csresnext50-panet-spp.cfg" % _ROOT_, help="path to model definition file")
    # parser.add_argument("--weights_path", type=str, default="%s/det/CSPresnext50_pcdd1024/yolov3_ckpt_step_349998-the_best_one.pth" % _ROOT_, help="path to weights file")
    parser.add_argument("--data_config", type=str, default="%s/config/pcdd1024.data" % _ROOT_, help="path to data config file")
    parser.add_argument("--class_path", type=str, default="%s/data/custom/classes.names" % _ROOT_, help="path to class label file")
    parser.add_argument("--folder", type=str, default="images")
    parser.add_argument("--label_folder", type=str, default="labels")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.3, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=8320, help="size of each image dimension")
    parser.add_argument("--overlap", type=int, default=0, help="size of each image dimension")
    parser.add_argument("--debug_mode", type=bool, default=False, help="size of each image dimension")
    parser.add_argument("--old_version", type=bool, default=False, help="size of each image dimension")
    parser.add_argument("--level", type=int, default=0, help="size of each image dimension")
    parser.add_argument("--center_side", type=int, default=20800, help="size of each image dimension")
    # parser.add_argument("--valid_path", type=str, default='%s/data_sets/sfy_3456789_slide_list.txt' % _ROOT_, help="size of each image dimension")
    parser.add_argument("--valid_path", type=str, default='%s/data_sets/visual_list.txt' % _ROOT_, help="size of each image dimension")
    # parser.add_argument("--valid_path", type=str, default='%s/data_sets/sfy_no_seq_manualcheck_list (xyw).txt' % _ROOT_, help="size of each image dimension")
    # parser.add_argument("--valid_path", type=str, default='%s/data_sets/sfy1&2_all_slide_list.txt' % _ROOT_, help="size of each image dimension")
    # parser.add_argument("--valid_path", type=str, default='%s/data_sets/sfy3456789_sequencesicnNewXYconversedisconserve0.txt' % _ROOT_, help="size of each image dimension")
    # parser.add_argument("--valid_path", type=str, default='%s/data_sets/sfy1&2_sequencesyolotinyNewXYconversedisconserve0.txt' % _ROOT_, help="size of each image dimension")
    parser.add_argument("--name", type=str, default='goldtest', help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    opt.valid_path = determinating_path(opt.valid_path)
    opt.model_def = determinating_path(opt.model_def)
    opt.weights_path = determinating_path(opt.weights_path)
    opt.data_config = determinating_path(opt.data_config)
    opt.class_path = determinating_path(opt.class_path)
    global isMNV2
    isMNV2 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    valid_path = opt.valid_path
    # valid_path = 'Z:/wei/PCDD_all/data_sets/sfy1&2_slide_goldtest_list.txt'
    # valid_path = 'Z:/wei/10X_srps/10X_srplist.txt'
    # valid_path = 'Z:/wei/10X_srps/10X_srplist.txt'
    # valid_path = 'W:/wei/PCDD_all/data_sets/sfy1&2_slide_val_list-for_160.txt'
    # Initiate model
    if not isMNV2:
        model = Darknet(opt.model_def, img_size=opt.img_size, lite_mode=True, use_final_loss=True, debug_mode=opt.debug_mode, old_version=opt.old_version).to(device)
        if opt.weights_path.endswith(".weights"):
            # Load darknet weights
            model.load_darknet_weights(opt.weights_path)
        else:
            # Load checkpoint weights
            model.load_state_dict(torch.load(opt.weights_path, map_location=device))
    else:
        weights_path = _ROOT0_ + "/cls/mobilenetv2_pretrained/checkpoint_best_oldtorch.pth"
        model = models.mobilenet.MobileNetV2(num_classes=2).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
    evaluate(
        model,
        list_path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
        fp_flag=True,
        images_folder=opt.folder,
        labels_folder=opt.label_folder,
        overlap=opt.overlap,
        level=opt.level,
        center_side=opt.center_side*2,
        name=opt.name
    )