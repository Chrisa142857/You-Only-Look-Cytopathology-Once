from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils.parse_config import *
from utils.utils import *
from utils.Mish.mish import Mish
from utils.FocalLoss import BCEFocalLoss
from utils.straightResModule import *

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_modules(module_defs, img_size=(1024, 1024), fp_max=0.5, use_final_loss=False, use_mish=False, use_fp_score=False, old_version=False, use_focal_loss=False, mode='det'):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    anchor_scale = (1, 1)
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            ###############
            if not old_version:
                if filters == 18:
                    filters = 21
                if filters == 12:
                    filters = 14
            ###############
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            # pad = int(module_def['pad'])
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                    groups=int(module_def['groups']) if 'groups' in module_def else 1,
                ),
            )


            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))
            elif module_def["activation"] == "mish":
                modules.add_module(f"leaky_{module_i}", Mish())

        elif module_def["type"] == "sResMB":
            bn = int(module_def["batch_normalize"]) if 'batch_normalize' in module_def else 0
            filters = int(module_def["filters"])
            pad = int(module_def['pad'])
            ###############
            if not old_version:
                if filters == 18:
                    filters = 21
                if filters == 12:
                    filters = 14
            ###############
            modules.add_module(
                f"sRMB_{module_i}",
                sResModuleBlock(
                    block_id=module_i,
                    ind_num=int(module_def["ind_num"]),
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    pad=pad,
                    stride=int(module_def["stride"]),
                    bn=bn,
                ),
            )

            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))
            elif use_mish:
                modules.add_module(f"leaky_{module_i}", Mish())

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample":
            if 'stride' in module_def:
                upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            elif 'nopad' in module_def:
                upsample = NopadUpsample(mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())

        elif module_def["type"] == "shortcut":
            filters = max(output_filters[1:][int(module_def["from"])], output_filters[-1])
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))
            elif module_def["activation"] == "mish":
                modules.add_module(f"leaky_{module_i}", Mish())

        elif module_def["type"] == "reorg3d":
            stride = int(module_def["stride"])
            filters = output_filters[-1]*stride*stride
            modules.add_module(f"reorg3d_{module_i}", EmptyLayer())

        elif module_def['type'] == 'crop':
            filters = output_filters[-1]
            modules.add_module(f"crop2d_{module_i}", EmptyLayer())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(int(anchors[i]*anchor_scale[0]), int(anchors[i + 1]*anchor_scale[1])) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, fp_max=fp_max, use_final_loss=use_final_loss, use_fp_score=use_fp_score, old_version=old_version, use_focal_loss=use_focal_loss, mode=mode)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)
    return hyperparams, module_list


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class NopadUpsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, mode="nearest"):
        super(NopadUpsample, self).__init__()
        self.mode = mode

    def forward(self, x, img_dim=None):
        img_dim = max(img_dim)
        scale_factor = (2*img_dim - 128) / (img_dim - 320)
        x = F.interpolate(x, scale_factor=scale_factor, mode=self.mode)
        return x


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, fp_max=0.5, use_final_loss=False, use_fp_score=False, old_version=False, use_focal_loss=False, mode='det'):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        if use_focal_loss:
            self.bce_loss = BCEFocalLoss()
        else:
            self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.fp_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.grid_size = 0  # grid size
        self.fp_max = fp_max
        self.use_final_loss = use_final_loss
        self.use_fp_score = use_fp_score
        self.fp_trained_flag = False
        self.old_version = old_version
        self.mode = mode

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g_x = self.grid_size[1]
        g_y = self.grid_size[0]
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride_w = self.img_dim[1] / g_x
        self.stride_h = self.img_dim[0] / g_y
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g_x).repeat(g_y, 1).view([1, 1, g_y, g_x]).type(FloatTensor)
        self.grid_y = torch.arange(g_y).repeat(g_x, 1).t().view([1, 1, g_y, g_x]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride_w, a_h / self.stride_h) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None, fp_flag=False):

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = [x.size(2), x.size(3)]
        if not self.old_version:
            prediction = (
                x.view(num_samples, self.num_anchors, self.num_classes + 6, grid_size[0], grid_size[1])
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )
        else:
            prediction = (
                x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size[0], grid_size[1])
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        # x = prediction[..., 0]  # Center x
        # y = prediction[..., 1]  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height

        pred_conf_sig = torch.sigmoid(prediction[..., 4])  # Conf
        pred_conf = prediction[..., 4]  # Conf
        # pred_fp_conf = torch.sigmoid(prediction[..., 5]) * self.fp_max  # Conf
        if not self.old_version:
            # pred_fp_conf = torch.sigmoid(prediction[..., 5])  # Conf
            pred_fp_conf = prediction[..., 5]  # Conf
            # pred_cls = torch.sigmoid(prediction[..., 6:])  # Cls pred.
            pred_cls = prediction[..., 6:]  # Cls pred.
            pred_final_conf = pred_conf * torch.exp(-1 * pred_fp_conf)
        else:
            # pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.
            pred_cls = prediction[..., 5:]  # Cls pred.
            pred_fp_conf = pred_conf.clone()
            pred_final_conf = pred_conf
        pred_final_conf_sig = torch.sigmoid(pred_final_conf)
        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = (x.data + self.grid_x) * self.stride_w
        pred_boxes[..., 1] = (y.data + self.grid_y) * self.stride_h
        pred_boxes[..., 2] = (torch.exp(w.data) * self.anchor_w) * self.stride_w
        pred_boxes[..., 3] = (torch.exp(h.data) * self.anchor_h) * self.stride_h

        if not targets is None:
            class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf, fp_mask, tfpconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=torch.sigmoid(pred_cls),
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
                use_fp_score=self.use_fp_score,
            )
            gt_isZero = False
            fp_isZero = False
            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_cls = nn.BCEWithLogitsLoss()(pred_cls[obj_mask], tcls[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf_sig[noobj_mask], tconf[noobj_mask])

            if x[obj_mask].shape == torch.Size([0]):
                gt_isZero = True
            if tfpconf[fp_mask].shape == torch.Size([0]):
                fp_isZero = True
                
            if not gt_isZero:
                loss_conf_obj = self.bce_loss(pred_conf_sig[obj_mask], tconf[obj_mask])
                loss_gt = self.noobj_scale * loss_conf_noobj + self.obj_scale * loss_conf_obj
            else:
                loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
                loss_gt = self.noobj_scale * loss_conf_noobj

            if not fp_isZero:
                loss_final_conf_obj = self.bce_loss(pred_final_conf_sig[fp_mask], tconf[fp_mask])
                loss_fp_conf_obj = nn.BCEWithLogitsLoss()(pred_fp_conf[fp_mask], tfpconf[fp_mask])
                self.fp_trained_flag = True
                if self.use_final_loss:
                    loss_fp = self.fp_scale * (loss_fp_conf_obj + loss_final_conf_obj)
                else:
                    loss_fp = self.fp_scale * loss_fp_conf_obj
                loss_conf = loss_gt + loss_fp
            else:
                loss_final_conf_obj = self.bce_loss(pred_final_conf_sig[fp_mask], tconf[fp_mask])
                loss_fp_conf_obj = nn.BCEWithLogitsLoss()(pred_fp_conf[fp_mask], tfpconf[fp_mask])
                self.fp_trained_flag = False
                loss_conf = loss_gt

            if not gt_isZero and self.mode == 'det':
                total_loss = loss_conf + loss_x + loss_y + loss_w + loss_h + loss_cls
            else:
                total_loss = loss_conf

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf_sig[obj_mask].mean()
            conf_noobj = pred_conf_sig[noobj_mask].mean()

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "loss_conf": to_cpu(loss_conf).item(),
                "loss_conf_obj": to_cpu(loss_conf_obj).item(),
                "loss_conf_noobj": to_cpu(loss_conf_noobj).item(),
                "loss_conf_fp": to_cpu(loss_fp_conf_obj).item(),
                "loss_conf_fpxgt": to_cpu(loss_final_conf_obj).item(),
                "loss_x": to_cpu(loss_x).item(),
                "loss_y": to_cpu(loss_y).item(),
                "loss_w": to_cpu(loss_w).item(),
                "loss_h": to_cpu(loss_h).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size_x": grid_size[0],
                "grid_size_y": grid_size[1],
            }
            if self.mode == 'det':

                output = torch.cat(
                    (
                        pred_boxes.view(num_samples, -1, 4),
                        pred_final_conf_sig.view(num_samples, -1, 1),
                        torch.sigmoid(pred_cls).view(num_samples, -1, self.num_classes),
                    ),
                    -1,
                )
            elif self.mode == 'cls':

                output = torch.cat(
                    (
                        pred_boxes.view(num_samples, -1, 4),
                        pred_final_conf_sig.view(num_samples, -1, 1),
                        tconf.view(num_samples, -1, 1),
                    ),
                    -1,
                )
            return output, total_loss
        else:
            if fp_flag:
                pred_final_conf = pred_conf * torch.exp(-1 * pred_fp_conf)
            else:
                pred_final_conf = pred_conf
            pred_final_conf_sig = torch.sigmoid(pred_final_conf)
            output = torch.cat(
                (
                    pred_boxes.view(num_samples, -1, 4),
                    pred_final_conf_sig.view(num_samples, -1, 1),
                    torch.sigmoid(pred_cls).view(num_samples, -1, self.num_classes),
                ),
                -1,
            )
            return output, 0


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=(1024, 1024), fp_max=0.5, use_final_loss=True, use_mish=False, use_fp_score=True, old_version=False, use_focal_loss=True, lite_mode=False, debug_mode=False, mode='det'):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs, img_size=img_size, fp_max=fp_max,
                                                            use_final_loss=use_final_loss, use_mish=use_mish,
                                                            use_fp_score=use_fp_score, old_version=old_version,
                                                            use_focal_loss=use_focal_loss, mode=mode)
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]
        self.img_size = img_size
        self.lite_mode = lite_mode
        self.debug_mode = debug_mode
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)
        self.rem_layers = []
        self.downSampler = torch.nn.AvgPool2d(2, stride=2)
        for i, module_def in enumerate(self.module_defs):
            if module_def["type"] == "route":
                for layer_i in module_def["layers"].split(","):
                    layer_i = int(layer_i)
                    if layer_i > 0:
                        self.rem_layers.append(layer_i)
                    else:
                        self.rem_layers.append(i + layer_i)
            elif module_def["type"] == "shortcut":
                self.rem_layers.append(i + int(module_def["from"]))
        # print(self.module_list)

    def forward(self, x, targets=None, fp_flag=False, x_pre_filters="12"):
        if x_pre_filters == "14": x_pre_filters = "12"
        if self.lite_mode is None: self.lite_mode = False
        if self.debug_mode is None: self.debug_mode = False
        img_dim = x.shape[2:4]
        d = max(img_dim)

        loss = 0
        layer_outputs, yolo_outputs, feature_maps, yolos = [], [], [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):

            if module_def["type"] == "convolutional":
                if module_def["filters"] == x_pre_filters:
                    feature_maps.append(x.clone().cpu())
            if module_def["type"] in ["convolutional", "maxpool", "sResMB"]:
                x = module(x)
            elif module_def['type'] == 'upsample':
                if 'nopad' in module_def:
                    stage4_size = int(d/16 - 4)
                    x = F.interpolate(x, size=stage4_size, mode='nearest')
                else:
                    x = module(x)
            elif module_def['type'] == 'crop':
                side = int(int(module_def['side'])/2)
                x = x[:,:, side:-side, side:-side]
            elif module_def["type"] == "reorg3d":
                x = reorg3d(x, int(module_def["stride"]))
            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1]
                a = layer_outputs[layer_i]
                if self.lite_mode:
                    if x == [] or a == []:
                        print("Opps, found BUG in rem_layers: ", self.rem_layers)
                        exit()
                nx = x.shape[1]
                na = a.shape[1]
                if nx == na:
                    x = x + a
                elif nx > na:
                    x[:, :na] = x[:, :na] + a
                else:
                    x = x + a[:, :nx]
                x = module(x)
            elif module_def["type"] == "yolo":
                yolos.append(x.clone().cpu())
                x, layer_loss = module[0](x, targets, img_dim, fp_flag=fp_flag)
                loss += layer_loss
                yolo_outputs.append(x.clone().cpu())
            if self.lite_mode:
                if i in self.rem_layers:
                    layer_outputs.append(x)
                else:
                    layer_outputs.append([])
            else:
                layer_outputs.append(x)
            if self.debug_mode:
                print(module_def['type'], ": ", x.shape)
                print("Press a key to continue...")
                input()
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        return {'output': yolo_outputs, 'feature_map': feature_maps, 'yolo': yolos} if targets is None else (loss, yolo_outputs)
        

    def load_darknet_weights(self, weights_path, use_torch=False, cutoff=None):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        if use_torch:
            x = torch.load(weights_path)
            model_dict = self.state_dict()
            weights = {}
            for i, key in enumerate(x):
                if i == cutoff:
                    break
                weights[key] = x[key]
                model_dict.update(weights)
                self.load_state_dict(model_dict)
            return
        else:
            with open(weights_path, "rb") as f:
                header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
                self.header_info = header  # Needed to write header when saving weights
                self.seen = header[3]  # number of images seen during training
                weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()
