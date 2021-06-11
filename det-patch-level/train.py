from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import det_evaluate, cls_evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from apex import amp
import torch.optim as optim
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3-sRMB-v02.cfg", help="path to model definition file")
    # parser.add_argument("--data_config", type=str, default="data/pcdd1024_all_new.data", help="path to data config file")
    parser.add_argument("--data_config", type=str, default="config/pcdd1024.data", help="path to data config file")
    #weights/yolov3_ckpt_darknet53_fp_init.pth
    #weights/darknet53.conv.74
    #darknet53_pcdd1024/yolov3_ckpt_step_349998.pth
    parser.add_argument("--pretrained_weights", type=str, default="Z:/wei/PyTorch-YOLOv3-master-orig/sRMBv02-cls/yolov3_ckpt_step_1649998.pth", help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=1024, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=10000, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=True, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--log_dir", default="sRMBv02-cls")
    parser.add_argument("--save_path", default="sRMBv02-cls")
    parser.add_argument("--mode", type=str, default="cls")
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    parser.add_argument("--lr_max_step", type=int, default=300000)
    parser.add_argument("--folder", type=str, default="train_images1024_new")
    parser.add_argument("--eval_folder", type=str, default="train_images1024_new")
    parser.add_argument("--fp_record", type=bool, default=False)
    parser.add_argument("--fp_interval", type=int, default=2)
    parser.add_argument("--neg_sample_interval", type=int, default=2)
    parser.add_argument("--fp_sample_interval", type=int, default=10)
    parser.add_argument("--start_step", type=int, default=1650000)
    parser.add_argument("--start_epoch", type=int, default=90)
    parser.add_argument("--use_fp_layer", type=bool, default=True)
    parser.add_argument("--use_final_loss", type=bool, default=True)
    parser.add_argument("--use_mish", type=bool, default=False) # ignore
    parser.add_argument("--use_fp_score", type=bool, default=True)
    parser.add_argument("--load_neck_only", type=bool, default=False)
    parser.add_argument("--fp_restrain", type=bool, default=True)
    parser.add_argument("--use_focal_loss", type=bool, default=True)
    parser.add_argument("--skip_first_train", type=bool, default=False)
    parser.add_argument("--group_conv", type=bool, default=False)
    parser.add_argument("--lite_mode", type=bool, default=True)
    parser.add_argument("--debug_mode", type=bool, default=False)
    ROOT = 'Z:/wei/PyTorch-YOLOv3-master-orig/'
    opt = parser.parse_args()
    print(opt)
    NEG_SAMPLE_INTERVAL = opt.neg_sample_interval
    FP_SAMPLE_INTERVAL = opt.fp_sample_interval
    if opt.mode == 'det':
        evaluate = det_evaluate
        eval_metric_name_1 = 'AP'
        eval_metric_name_2 = 'f1'
    elif opt.mode == 'cls':
        evaluate = cls_evaluate
        eval_metric_name_1 = 'iou'
        eval_metric_name_2 = 'dice'
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    log_dir = ROOT+opt.log_dir + '/log-'+str(datetime.datetime.now()).replace(':', '-')
    os.makedirs(ROOT+opt.save_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    orig_img_size = (opt.img_size, opt.img_size)
    if opt.img_size == 1936:
        orig_img_size = (1216, 1936)
    logger = Logger(log_dir)
    with open(log_dir+'.txt', 'w') as log_txt:
        log_txt.write(str(opt) + '\n')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(ROOT+opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(ROOT+opt.model_def, img_size=orig_img_size, use_final_loss=opt.use_final_loss, use_mish=opt.use_mish, use_fp_score=opt.use_fp_score, use_focal_loss=opt.use_focal_loss, lite_mode=opt.lite_mode, debug_mode=opt.debug_mode, mode=opt.mode).to(device)
        
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if not opt.load_neck_only:
            if opt.pretrained_weights.endswith(".pth"):
                pretrained_dict = torch.load(opt.pretrained_weights)
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} 
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
            else:
                model.load_darknet_weights(opt.pretrained_weights)
        else:
            model.load_darknet_weights(opt.pretrained_weights, use_torch=True, cutoff=318)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    # model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.lr_max_step/opt.batch_size, eta_min=1e-8)
    # model = nn.DataParallel(model, device_ids=[0,1]) # multi-GPU
    metrics = [
        "loss",
        "loss_conf",
        "loss_conf_obj",
        "loss_conf_noobj",
        "loss_conf_fp",
        "loss_conf_fpxgt",
        "loss_x",
        "loss_y",
        "loss_w",
        "loss_h",
        "cls",
        "cls_acc",
        "conf_obj",
        "conf_noobj",
        "grid_size_x",
        "grid_size_y",
    ]

    fp_dict = {}
    batches_done = opt.start_step
    fp_record_times = 0
    for iii in range(0, batches_done):
        if iii % opt.gradient_accumulations:
            scheduler.step()
    for epoch in range(opt.start_epoch, opt.epochs):
        model.train()
        fp_nums = []
        # ------------------------------------------------------
        #   [Begin] Random Augmented / Normal training process
        # ------------------------------------------------------        dataset = ListDataset("Z:/wei/PCDD/data_sets/images1536_valid.txt", img_size=(1536,1536), folder="train_images1536", augment=True, multiscale=True)
        if not (epoch == opt.start_epoch and opt.skip_first_train):
            dataset = ListDataset(train_path, img_size=orig_img_size, folder=opt.folder, augment=True, multiscale=opt.multiscale_training, fp_dict=fp_dict)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.n_cpu,
                pin_memory=True,
                collate_fn=dataset.collate_fn,
            )
            start_time = time.time()
            neg_sample_times = 0
            for batch_i, (img_path, imgs, targets) in enumerate(dataloader):
                train_flag = True
                if targets.shape[0] == 0:
                    train_flag = neg_sample_times % NEG_SAMPLE_INTERVAL == 0
                    neg_sample_times += 1
                if not train_flag:
                    continue
                batches_done += 1
                imgs = Variable(imgs.to(device), requires_grad=True)
                targets = Variable(targets, requires_grad=False).to(device)
                new_img_size = (imgs.shape[3], imgs.shape[2])
                # print(new_img_size, dataset.img_size)
                #############################################
                # print(new_img_size, imgs.shape)
                # print(new_img_size, targets)
                # import matplotlib.pyplot as plt
                # numpy_img = to_cpu(imgs[0]).numpy().transpose((1,2,0))
                # plt.imshow(numpy_img)
                # for i_i, (x,y,w,h) in enumerate(targets[:,2:]):
                #     if targets[i_i, 0] == 0:
                #         plt.gca().add_patch(
                #             plt.Rectangle(((x-w/2)*new_img_size[0], (y-h/2)*new_img_size[1]), width=w*new_img_size[0], height=h*new_img_size[1], fill=False)
                #             )
                # plt.show()
                # continue
                #############################################
                loss, _ = model(imgs, targets)
                loss.backward()
                if batches_done % opt.gradient_accumulations:
                    # Accumulates gradient before each step
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                # ----------------
                #   Log progress
                # ----------------
                log_str = "\n---- [Epoch %d/%d, Batch %d/%d, Pure Training] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))
                log_str += f"\nImage size {new_img_size}\n"
                metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

                # Log metrics at each YOLO layer
                for i, metric in enumerate(metrics):
                    formats = {m: "%.6f" for m in metrics}
                    formats["grid_size_x"] = "%2d"
                    formats["grid_size_y"] = "%2d"
                    formats["cls_acc"] = "%.2f%%"
                    row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                    metric_table += [[metric, *row_metrics]]

                    # Tensorboard logging
                    tensorboard_log = []
                    for j, yolo in enumerate(model.yolo_layers):
                        for name, metric in yolo.metrics.items():
                            if name != "grid_size_x" and name != "grid_size_y":
                                tensorboard_log += [(f"{name}_{j+1}", metric)]
                    tensorboard_log += [("loss", loss.item())]
                    tensorboard_log += [("lr", scheduler.get_lr()[0])]
                    logger.list_of_scalars_summary(tensorboard_log, batches_done)

                log_str += AsciiTable(metric_table).table
                log_str += f"\nTotal loss {loss.item()}"
                log_str += f"\nLearning rate {scheduler.get_lr()[0]}"


                # Determine approximate time left for epoch
                epoch_batches_left = len(dataloader) - (batch_i + 1)
                time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
                log_str += f"\n---- ETA {time_left}"


                if (batches_done + NEG_SAMPLE_INTERVAL) % opt.checkpoint_interval == 0:
                    torch.save(model.state_dict(), f"%s/yolov3_ckpt_step_%d.pth" % (ROOT+opt.save_path, batches_done))
                    print("\n---- Evaluating Model ----")
                    # Evaluate the model on the validation set
                    precision, recall, AP, f1, ap_class, val_fp_nums, mloss = evaluate(
                        model,
                        path=valid_path,
                        iou_thres=0.5,
                        conf_thres=0.5,
                        nms_thres=0.5,
                        img_size=orig_img_size,
                        batch_size=opt.batch_size,
                        fp_flag=fp_record_times > 0 or opt.use_fp_layer,
                        images_folder=opt.eval_folder,
                        step=batches_done,
                        name=opt.model_def[:-4].split('-')[-1],
                        NEG_SAMPLE_INTERVAL=NEG_SAMPLE_INTERVAL,
                    )
                    evaluation_metrics = [
                        ("step_val_precision", precision.mean()),
                        ("step_val_recall", recall.mean()),
                        ("step_val_%s" % eval_metric_name_1, AP.mean()),
                        ("step_val_%s" % eval_metric_name_2, f1.mean()),
                        ("step_train_mFP_num", np.mean(fp_nums)),
                        ("step_val_mFP_num", np.mean(val_fp_nums)),
                        ("step_val_loss", mloss.item()),
                        ("fp_record_times", fp_record_times),
                    ]
                    logger.list_of_scalars_summary(evaluation_metrics, batches_done)
                    log_str += "\n---- [Epoch %d/%d, Evaluation] ----\n" % (epoch, opt.epochs)
                    log_str += f"\nVal loss {mloss.item()}"
                    log_str += f"\nVal {eval_metric_name_1} {AP.mean()}"
                    log_str += f"\nVal {eval_metric_name_2} {f1.mean()}"
                    log_str += f"\nVal precision {precision.mean()}"
                    log_str += f"\nVal recall {recall.mean()}"

                print(log_str)

                with open(log_dir + '.txt', 'a') as log_txt:
                    log_txt.write(log_str + '\n')
                model.seen += imgs.size(0)
        # ------------------------------------------------------
        #   [End] Random Augmented / Normal training process
        # ------------------------------------------------------

        # --------------------------------------
        #   [Begin] Training with FPs recording
        # --------------------------------------
        if opt.fp_record and (epoch + 1) % opt.fp_interval == 0:
            fp_record_times += 1
            dataset = ListDataset(train_path, img_size=orig_img_size, folder=opt.folder, augment=False, multiscale=opt.multiscale_training, fp_dict=fp_dict)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.n_cpu,
                pin_memory=False,
                collate_fn=dataset.collate_fn,
            )
            start_time = time.time()
            fp_sample_times = 0
            neg_sample_times = 0
            for batch_i, (img_path, imgs, targets) in enumerate(dataloader):
                imgs = Variable(imgs.to(device))
                _gt = Variable(targets.clone())
                targets = Variable(targets, requires_grad=False).to(device)

                new_img_size = (imgs.shape[3], imgs.shape[2])
                train_flag = True
                if targets.shape[0] == 0:
                    train_flag = neg_sample_times % NEG_SAMPLE_INTERVAL == 0
                    neg_sample_times += 1
                if train_flag:
                    model.train()
                    loss, outputs = model(imgs, targets, fp_flag=fp_record_times > 1 or opt.use_fp_layer)
                    batches_done += 1
                    loss.backward()
                else:
                    model.eval()
                    outputs = model(imgs, None, fp_flag=fp_record_times > 1 or opt.use_fp_layer)

                if (batches_done + NEG_SAMPLE_INTERVAL) % opt.checkpoint_interval == 0:
                    torch.save(model.state_dict(), f"%s/yolov3_ckpt_step_%d.pth" % (ROOT+opt.save_path, batches_done))
                    print("\n---- Evaluating Model ----")
                    # Evaluate the model on the validation set
                    precision, recall, AP, f1, ap_class, val_fp_nums, mloss = evaluate(
                        model,
                        path=valid_path,
                        iou_thres=0.5,
                        conf_thres=0.5,
                        nms_thres=0.5,
                        img_size=orig_img_size,
                        batch_size=opt.batch_size,
                        fp_flag=fp_record_times > 0 or opt.use_fp_layer,
                        images_folder=opt.eval_folder,
                        step=batches_done,
                        name=opt.model_def[:-4].split('-')[-1],
                        NEG_SAMPLE_INTERVAL=NEG_SAMPLE_INTERVAL,
                    )
                    evaluation_metrics = [
                        ("step_val_precision", precision.mean()),
                        ("step_val_recall", recall.mean()),
                        ("step_val_%s" % eval_metric_name_1, AP.mean()),
                        ("step_val_%s" % eval_metric_name_2, f1.mean()),
                        ("step_train_mFP_num", np.mean(fp_nums)),
                        ("step_val_mFP_num", np.mean(val_fp_nums)),
                        ("fp_record_times", fp_record_times),
                    ]
                    logger.list_of_scalars_summary(evaluation_metrics, batches_done)

                outputs = non_max_suppression(outputs, conf_thres=0.5, nms_thres=0.5, max_thres=0.5+(0.05*fp_record_times))
                # Rescale _gt
                _gt = _gt[torch.where(_gt[:, 1]==0)[0], :]
                _gt[:, 2:] = xywh2xyxy(_gt[:, 2:])
                _gt[:, 2] *= new_img_size[0]
                _gt[:, 3] *= new_img_size[1]
                _gt[:, 4] *= new_img_size[0]
                _gt[:, 5] *= new_img_size[1]
                restrain_flag = True
                if opt.fp_restrain:
                    restrain_flag = fp_sample_times % FP_SAMPLE_INTERVAL == 0
                    fp_sample_times += 1
                    # restrain_flag = _gt.shape[0] == 0 and fp_sample_times % FP_SAMPLE_INTERVAL == 0
                    # if _gt.shape[0] == 0:
                    #     fp_sample_times += 1
                if restrain_flag:
                    # collect FPs
                    sample_metrics = get_batch_statistics(outputs, _gt, iou_threshold=0.1)
                    po = [o for o in outputs if o is not None]
                    for path, _o, sample_metric in zip(img_path, po, sample_metrics):
                        true_positives, _, _ = sample_metric
                        fp_id = np.where(true_positives == 0)[0]
                        if len(true_positives) != _o.shape[0]:
                            print('Opp, BUG in fp finding: ', fp_id, _o.shape)
                            exit()

                        fp_dict[path] = torch.cat((_o[fp_id, 4:5]*-1, xyxy2xywh(_o[fp_id, :4], is_scaled=False, img_size=new_img_size)), 1)
                        fp_nums.append(len(fp_id))
                if batches_done % opt.gradient_accumulations:
                    # Accumulates gradient before each step
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                if train_flag:
                    # ----------------
                    #   Log progress
                    # ----------------
                    log_str = "\n---- [Epoch %d/%d, Batch %d/%d, Training with FPs recording] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))
                    log_str += f"\nImage size {new_img_size}\n"
                    metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

                    # Log metrics at each YOLO layer
                    for i, metric in enumerate(metrics):
                        formats = {m: "%.6f" for m in metrics}
                        formats["grid_size_x"] = "%2d"
                        formats["grid_size_y"] = "%2d"
                        formats["cls_acc"] = "%.2f%%"
                        row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                        metric_table += [[metric, *row_metrics]]

                        # Tensorboard logging
                        tensorboard_log = []
                        for j, yolo in enumerate(model.yolo_layers):
                            for name, metric in yolo.metrics.items():
                                if name != "grid_size":
                                    tensorboard_log += [(f"{name}_{j+1}", metric)]
                        tensorboard_log += [("loss", loss.item())]
                        tensorboard_log += [("lr", scheduler.get_lr()[0])]
                        logger.list_of_scalars_summary(tensorboard_log, batches_done)

                    log_str += AsciiTable(metric_table).table
                    log_str += f"\nTotal loss {loss.item()}"
                    log_str += f"\nLearning rate {scheduler.get_lr()[0]}"

                    # Determine approximate time left for epoch
                    epoch_batches_left = len(dataloader) - (batch_i + 1)
                    time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
                    log_str += f"\n---- ETA {time_left}"

                    print(log_str)

                    model.seen += imgs.size(0)
        # --------------------------------------
        #   [End] Training with FPs recording
        # --------------------------------------

        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class, val_fp_nums, mloss = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=orig_img_size,
                batch_size=opt.batch_size,
                fp_flag=fp_record_times > 0 or opt.use_fp_layer,
                images_folder=opt.eval_folder,
                step=batches_done,
                name=opt.model_def[:-4].split('-')[-1],
                NEG_SAMPLE_INTERVAL=NEG_SAMPLE_INTERVAL,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_%s" % eval_metric_name_1, AP.mean()),
                ("val_%s" % eval_metric_name_2, f1.mean()),
                ("train_mFP_num", np.mean(fp_nums)),
                ("val_mFP_num", np.mean(val_fp_nums)),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name",  eval_metric_name_1]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- %s {AP.mean()}" % eval_metric_name_1)
            print(f"---- val_mFP_num {np.mean(val_fp_nums)}")
            print(f"---- train_mFP_num {np.mean(fp_nums)}")
            torch.save(model.state_dict(), f"%s/yolov3_ckpt_epoch_%d.pth" % (ROOT+opt.save_path, epoch))

