from model import create_model
from dataset import SequenceDataset, MaxpoolDataset, ImgDataset
from tqdm import tqdm

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from utils import calc_err, Logger, repackage_hidden
from inference import inference
from TBPTT import TBPTT
_ROOT_ = 'D:\\WSI_analysis\\rnn'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', default='2', type=str, help='gpu device id')
    parser.add_argument('--test-every', default=1, type=int, help='test_every')
    parser.add_argument('--batch-size', default=32, type=int, help='batch_size')
    parser.add_argument('--workers', default=2, type=int, help='workers')
    parser.add_argument('--epochs', default=300, type=int, help='epochs')
    # parser.add_argument('--train-list', default=r'%s\data_sets\sfy1&2_all_slide_list_MILtrain.txt'%_ROOT_, type=str)
    # parser.add_argument('--val-list', default=r'%s\data_sets\sfy1&2_all_slide_list_MILval.txt'%_ROOT_, type=str)
    parser.add_argument('--train-list', default=r'%s\data_sets\sfy_all_slide_list_train.txt'%_ROOT_, type=str)
    parser.add_argument('--val-list', default=r'%s\data_sets\sfy_all_slide_list_val.txt'%_ROOT_, type=str)
    # parser.add_argument('--train-list', default=r'%s\data_sets\sfy_all_slide_train_formanualcheck.txt'%_ROOT_, type=str)
    # parser.add_argument('--val-list', default=r'%s\data_sets\manualcheck_121slide_list.txt'%_ROOT_, type=str)
    # parser.add_argument('--train-list', default=r'%s\data_sets\sfy_3456789_slide_list_train.txt'%_ROOT_, type=str)
    # parser.add_argument('--val-list', default=r'%s\data_sets\sfy_3456789_slide_list_val.txt'%_ROOT_, type=str)
    parser.add_argument('--output-dir', default=r'%s_output'%_ROOT_, type=str)
    parser.add_argument('--lr', default=5e-6, type=float)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--resume-epochs', default=100, type=int)
    parser.add_argument('--model', default='transformer_yolotinysfyalldisconverse0top200', type=str)
    parser.add_argument('--n-layers', default=10, type=int)
    parser.add_argument('--hidden-dim', default=2048, type=int)
    parser.add_argument('--top-n', default=200, type=int)
    parser.add_argument('--bottom-n', default=0, type=int)
    parser.add_argument('--dis-thres', default=0, type=float)
    parser.add_argument('--joint', default=0, type=float)
    parser.add_argument('--lr-scheduler', default=True, type=bool)
    parser.add_argument('--balance', default=False, type=bool)
    parser.add_argument('--tbptt', default=False, type=bool)
    parser.add_argument('--name', default='yolotinydisconserve0', type=str, help='default | yolo_conf_fpconf | conf_distanceconserve | conf_fpconf | old | yolov3old | disx | old_xyconverse')
    parser.add_argument('--data-type', default='seq', type=str, help='seq | img | maxpool')
    parser.add_argument('--fm-root', default='', type=str, help=' | yolo ')
    parser.add_argument('--data-root', default='sfyall', type=str, help='sfy1&2 | sfy3456789 | sfyall')


    args = parser.parse_args()
    os.makedirs(os.path.join(args.output_dir, 'logs', 'v4_%s' % args.model), exist_ok=True)
    logger = Logger(os.path.join(args.output_dir, 'logs', 'v4_%s' % args.model))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    if args.bottom_n > 10000 or args.top_n > 10000:
        torch.backends.cudnn.enabled = False
    if 'mnv2' in args.name:
        input_dim = 1280
    elif 'yolo' not in args.name:
        input_dim = 768
    elif 'yolotiny' in args.name:
        input_dim = 768
    else:
        if 'yolov3' in args.name and 'New' in args.name:
            input_dim = 1792
        elif 'yolov3' in args.name:
            input_dim = 1536
        else:
            input_dim = 42
    model = create_model(args.model)(input_dim=input_dim, batch_size=args.batch_size, n_layers=args.n_layers, hidden_dim=args.hidden_dim).cuda()
    lr = args.lr
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if args.data_type == 'seq':
        PthDataset = SequenceDataset
    elif args.data_type == 'img':
        PthDataset = ImgDataset
    elif args.data_type == 'maxpool':
        PthDataset = MaxpoolDataset

    print('Train Data')
    trainData = PthDataset(r'F:\sfy1&2_2wfm%s'%args.fm_root, args.train_list, r'D:\sfy1&2_yolos', name=args.name,
                           top_n=args.top_n, bottom_n=args.bottom_n, balance=args.balance, dis_thers=args.dis_thres, joint=args.joint, data_root=args.data_root)
    print('Valid Data')
    validData = PthDataset(r'F:\sfy1&2_2wfm%s'%args.fm_root, args.val_list, r'D:\sfy1&2_yolos', name='%s' % args.name,
                           top_n=args.top_n, bottom_n=args.bottom_n, balance=args.balance, dis_thers=args.dis_thres, joint=args.joint, data_root=args.data_root)
    if 'manualcheck' in args.val_list:
        with open(r'D:\WSI_analysis\rnn\data_sets\manualcheck_121slide_label.txt', 'r') as f:
            lines = f.read().split('\n')[:-1]
        with open(r'D:\WSI_analysis\rnn\data_sets\manualcheck_121slide_list.txt', 'r') as f:
            slides = f.read().split('\n')[:-1]
        manual_label={}
        for l, s in zip(lines, slides):
            manual_label[s.split('\\')[-1].replace('.srp','')] = int(l)
        for k in validData.slide2label:
            validData.slide2label[k] = manual_label[k]
    if args.lr_scheduler:
        lr_s = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(args.epochs*len(trainData)/args.batch_size), T_mult=2)
    else:
        lr_s = None
    train_loader = torch.utils.data.DataLoader(
        trainData,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False
    )
    val_loader = torch.utils.data.DataLoader(
        validData,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False
    )

    if args.resume:
        nepochs = args.epochs + args.resume_epochs
        resume = torch.load(os.path.join(args.output_dir, 'v4_%s_checkpoint_best.pth' % args.model))
        model.load_state_dict(resume['state_dict'])
        optimizer.load_state_dict(resume['optimizer'])
        start_epoch = resume['epoch'] - 1
        if lr_s is not None:
            for _ in range(start_epoch):
                for _ in range(len(train_loader)):
                    lr_s.step()
    else:
        nepochs = args.epochs
        start_epoch = 0

    best_acc = 0
    for epoch in tqdm(range(start_epoch, nepochs), desc='Start Epoch:'):
        loss = train(train_loader, model, criterion, optimizer, epoch, nepochs, args.output_dir, logger, args, lr_s, lr, tbptt=args.tbptt)
        print('Training\tEpoch: [%d/%d]\tLoss: %.6f' % (epoch + 1, nepochs, loss))

        if (epoch+1) % args.test_every == 0:
            err, fpr, fnr, best_acc = valid(epoch, val_loader, model, nepochs, validData, logger, best_acc, optimizer, args)


def train(train_loader, model, criterion, optimizer, run, nepochs, output_dir, logger, args, lr_s, lr, clip=5, tbptt=True):
    train_loader.dataset.setSlides()
    model.train()
    hidden = model.init_hidden()
    running_loss = 0.
    _tbptt = TBPTT(one_step_module=model, loss_module=criterion)
    for i, one in tqdm(enumerate(train_loader), desc='Training\tEpoch: [%d/%d]\t' % (run+1, nepochs)):
        data = one['data']
        target = one['label']
        model.zero_grad()
        if not tbptt:
            if 'transformer' not in args.model:
                output, hidden = model(data.cuda(), hidden)
                hidden = repackage_hidden(hidden)
            else:
                output = model(data.cuda())
            loss = criterion(output, target.float().cuda())
            loss.backward(retain_graph=True)
        else:
            loss = _tbptt.train(input_sequence=data.cuda(), targets=target, init_state=hidden)
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if lr_s is not None:
            lr_s.step()
            lr = lr_s.get_lr()[0]
        running_loss += loss.item()*data.size(0)

        tensorboard_log = [("loss", loss.item()), ('lr', lr)]
        logger.list_of_scalars_summary(tensorboard_log, i+run*len(train_loader))
    return running_loss/len(train_loader.dataset)


def valid(epoch, val_loader, model, nepochs, validData, logger, best_acc, optimizer, args, conf_thres=0.5, save_weight=True):
    val_loader.dataset.setSlides()
    probs, slides = inference(epoch, val_loader, model, nepochs, model_name=args.model)
    pred = {}
    for key in probs:
        pred[key] = 1 if probs[key] >= conf_thres else 0
    err, fpr, fnr = calc_err(pred, validData.slide2label, slides)
    output = err
    print('Validation\tEpoch: [%d/%d]\tError: %.3f\tFPR: %.3f\tFNR: %.3f' % (epoch + 1, nepochs, err, fpr, fnr))
    tensorboard_log = [("Error", err), ("FP ratio", fpr), ("FN ratio", fnr)]
    if logger is not None:
        logger.list_of_scalars_summary(tensorboard_log, epoch)
    # Save best model
    # err = (fpr + fnr) / 2.
    if 1 - err >= best_acc:
        best_acc = 1 - err
        if save_weight:
            obj = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict()
            }
            torch.save(obj, os.path.join(args.output_dir, 'v4_%s_checkpoint_best.pth' % args.model))
    return output, fpr, fnr, best_acc


if __name__ == "__main__":
    main()
