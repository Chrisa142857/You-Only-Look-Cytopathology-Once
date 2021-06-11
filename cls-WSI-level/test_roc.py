from model import create_model
from dataset import SequenceDataset
from train import valid
from inference import inference
from utils import calc_err

import os
import torch
import argparse
_ROOT_ = 'D:\\WSI_analysis\\rnn'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', default='2', type=str, help='gpu device id')
    parser.add_argument('--test-every', default=1, type=int, help='test_every')
    parser.add_argument('--batch-size', default=32, type=int, help='batch_size')
    parser.add_argument('--workers', default=2, type=int, help='workers')
    parser.add_argument('--epochs', default=300, type=int, help='epochs')
    # parser.add_argument('--train-list', default=r'%s\data_sets\sfy_all_slide_list_train.txt'%_ROOT_, type=str)
    parser.add_argument('--val-list', default=r'%s\data_sets\sfy_all_slide_list_val.txt'%_ROOT_, type=str)
    # parser.add_argument('--train-list', default=r'%s\data_sets\sfy1&2_all_slide_list_MILtrain.txt'%_ROOT_, type=str)
    # parser.add_argument('--val-list', default=r'%s\data_sets\sfy1&2_all_slide_list_MILval.txt'%_ROOT_, type=str)
    # parser.add_argument('--val-list', default=r'%s\data_sets\manualcheck_42slide_list.txt'%_ROOT_, type=str)
    # parser.add_argument('--train-list', default=r'%s\data_sets\sfy_3456789_slide_list_train.txt'%_ROOT_, type=str)
    # parser.add_argument('--val-list', default=r'%s\data_sets\sfy_3456789_slide_list_val.txt'%_ROOT_, type=str)
    parser.add_argument('--output-dir', default=r'%s_output'%_ROOT_, type=str)
    parser.add_argument('--lr', default=5e-6, type=float)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--resume-epochs', default=100, type=int)
    # parser.add_argument('--model', default='transformer_icnNewXYsfy12disconverse0top100', type=str)
    # parser.add_argument('--model', default='transformer_icnsfyalldisconverse0top200', type=str)
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
    parser.add_argument('--data-root', default='sfyall', type=str, help='sfy1&2 | sfy3456789 | sfyall | manualcheck')
    parser.add_argument('--tag', default='', type=str)
    parser.add_argument('--infer-once', default=False, type=bool)

    args = parser.parse_args()
    # tag = '_manualcheck_42slide_retrained'
    tag = args.tag
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
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

    resume = torch.load(os.path.join(args.output_dir, 'v4_%s_checkpoint_best.pth' % args.model))
    model.load_state_dict(resume['state_dict'])

    print('Valid Data')
    validData = SequenceDataset(r'F:\sfy1&2_2wfm%s'%args.fm_root, args.val_list, r'D:\sfy1&2_yolos', name='%s' % args.name,
                           top_n=args.top_n, bottom_n=args.bottom_n, balance=args.balance, dis_thers=args.dis_thres, joint=args.joint, data_root=args.data_root)
    if 'manualcheck' in args.val_list:
        with open(args.val_list.replace('list', 'label'), 'r') as f:
            lines = f.read().split('\n')[:-1]
        with open(args.val_list, 'r') as f:
            slides = f.read().split('\n')[:-1]
        manual_label={}
        for l, s in zip(lines, slides):
            manual_label[s.split('\\')[-1].replace('.srp','')] = int(l)
        for k in validData.slide2label:
            validData.slide2label[k] = manual_label[k]
    val_loader = torch.utils.data.DataLoader(
        validData,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False
    )
    csv_name = args.model+tag
    if not args.infer_once:
        fconv = open(os.path.join(args.output_dir, 'v4_%s_ROC_val.csv' % csv_name), 'w')
        fconv.write('FPR,TPR\n')
        fconv.close()
    best_acc = 0
    best_res = []
    val_loader.dataset.setSlides()
    probs, slides = inference(0, val_loader, model, 1, model_name=args.model)
    with open(os.path.join(args.output_dir, 'v4_%s_pred_val.csv' % csv_name), 'w') as f:
        f.write('probs,labels\n')
    for key in probs:
        with open(os.path.join(args.output_dir, 'v4_%s_pred_val.csv' % csv_name), 'a') as f:
            f.write('%f,%f\n'% (probs[key], validData.slide2label[key]))
    for i, thres in enumerate(range(101)):
        thres /= 100
        pred = {}
        for key in probs:
            pred[key] = 1 if probs[key] >= thres else 0
        err, fpr, fnr = calc_err(pred, validData.slide2label, slides)
        print('Error: %.3f\tFPR: %.3f\tFNR: %.3f' % (err, fpr, fnr))
        acc = 1 - err
        if acc >= best_acc: 
            best_acc = acc
            best_res = [fpr, fnr, acc]
        fconv = open(os.path.join(args.output_dir, 'v4_%s_ROC_val.csv' % csv_name), 'a')
        fconv.write('%f,%f\n' % (fpr, 1-fnr))
        fconv.close()
        if args.infer_once: return
    if not args.infer_once:
        fconv = open(os.path.join(args.output_dir, 'v4_%s_ROC_val.csv' % csv_name), 'a')
        fconv.write('best acc, fpr, fnr\n')
        fconv.write('%f,%f,%f\n' % (best_acc, best_res[0], best_res[1]))
        fconv.close()


if __name__ == "__main__":
    main()
