from cls_train import prepare_tdata, random_sample_xy
from det_models import Darknet
import sys
import os
import cv2
import numpy as np
import argparse
import random
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from slide_readtool import fast_read
from sklearn.metrics import roc_curve, auc
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_def = "D:/WSI_analysis/det/config/yolov3-sRMB-v02.cfg"
val_lib = "D:/WSI_analysis/cls/data/det-valid.pth"


parser = argparse.ArgumentParser(description='MIL-nature-medicine-2019 tile classifier training script')
parser.add_argument('--output', type=str, default='icn_cls', help='name of output file')
parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size (default: 512)')
parser.add_argument('--nepochs', type=int, default=1, help='number of epochs')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')


img_size = 320
def main(model, infer_fn, is_norm=False):
    global args
    args = parser.parse_args()
    if is_norm:
    	normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    	trans = transforms.Compose([transforms.ToTensor(), normalize])
    else:
    	trans = transforms.Compose([transforms.ToTensor()])
    accs, tprs, tnrs = [], [], []
    aucs = []
    for _ in range(10):
        val_dset = CLSdataset(libraryfile=val_lib, transform=trans, name='valid')
        val_loader = torch.utils.data.DataLoader(
            val_dset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False, collate_fn=val_dset.collect_fn)
        val_dset.shuffledata()
        probs, targets = infer_fn(0, val_loader, model)
        pred = [1 if x >= 0.5 else 0 for x in probs]
        err,fpr,fnr = calc_err(pred, targets)
        accs.append(((1-fpr)+(1-fnr))/2)
        tprs.append(1-fpr)
        tnrs.append(1-fnr)
        fp,tp,threshold = roc_curve([t.int() for t in targets], pred)
        aucs.append(auc(fp, tp))
    return accs, tprs, tnrs, aucs

def calc_err(pred,real):
    pred = np.array(pred)
    real = np.array(real)
    neq = np.not_equal(pred, real)
    err = float(neq.sum())/pred.shape[0]
    fpr = float(np.logical_and(pred==1,neq).sum())/(real==0).sum()
    fnr = float(np.logical_and(pred==0,neq).sum())/(real==1).sum()
    return err, fpr, fnr


def inference_icn(run, loader, model):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    targets = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            print('Inference\tEpoch: [{}/{}]\tBatch: [{}/{}]'.format(run+1, args.nepochs, i+1, len(loader)))
            input = input.cuda()
            # stage4 = torch.max(model(input)['output'][:, :300, 4:5], dim=1)[0]
            # stage5 = torch.max(model(input)['output'][:, 300:, 4:5], dim=1)[0]
            # yolo_out = (stage4 + stage5) / 2
            yolo_out = torch.max(model(input)['output'][:, :, 4:5], dim=1)[0]
            probs[i*args.batch_size:i*args.batch_size+input.size(0)] = yolo_out.detach()[:,0].clone()
            targets[i*args.batch_size:i*args.batch_size+input.size(0)] = target.clone()
    return probs.cpu().numpy(), targets


def inference_mobilenet(run, loader, model):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    targets = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            print('Inference\tEpoch: [{}/{}]\tBatch: [{}/{}]'.format(run+1, args.nepochs, i+1, len(loader)))
            input = input.cuda()
            output = F.softmax(model(input), dim=1)
            probs[i*args.batch_size:i*args.batch_size+input.size(0)] = output.detach()[:,1].clone()
            targets[i*args.batch_size:i*args.batch_size+input.size(0)] = target.clone()
    return probs.cpu().numpy(), targets


class CLSdataset(data.Dataset):
   def __init__(self, libraryfile='', name='train', transform=None, patch_size=320, roi=2048):
       self.ann = torch.load(libraryfile)
       self.slide2IDX, slides = {}, []
       for i,s in enumerate(self.ann.keys()):
           self.slide2IDX[s] = i
           slides += [s]
       self.output = args.output
       self.name = name
       self.slides = slides
       self.transform = transform
       self.size = patch_size
       self.roi = roi
       self.level = 0
       self.recent_slide = ''
       self.read_class = {}
       self.images, self.images_index = {}, {}
   def shuffledata(self):
       self.visualized = 0
       self.targets = []
       self.init_tdatadict = prepare_tdata(self.slide2IDX, self.ann, self.size, self.roi)
       self.tdatadict, self.t_data, self.targets, self.xy2target = {}, [], [], {}
       for x in self.init_tdatadict:
        self.tdatadict[x] = random.sample(self.init_tdatadict[x], len(self.init_tdatadict[x]))
        self.xy2target[x] = {}
        for px, py, t in self.tdatadict[x]:
            self.xy2target[x]['%d,%d' % (int(px), int(py))] = t
        self.tdatadict[x] = [[d[0],d[1]] for d in self.tdatadict[x]]
        self.t_data += [x] * len(self.tdatadict[x])
   def init_read_class(self, img_size, LeftTopPts, level=0):
       self.read_class[self.recent_slide] = fast_read.ReadClass(self.recent_slide, False)
       self.read_class[self.recent_slide].setReadLevel(level)
       self.read_class[self.recent_slide].passRects([fast_read.Rect(int(x), int(y), int(img_size),int(img_size)) for x, y in LeftTopPts])
   def __getitem__(self, index):
       slideIDX = self.t_data[index]
       if slideIDX not in self.images_index: self.images_index[slideIDX] = 0
       if slideIDX not in self.images: self.images[slideIDX] = []
       if self.recent_slide != self.slides[slideIDX]:
           print('######################################')
           print('Init', self.slides[slideIDX])
           print('######################################')
           self.recent_slide = self.slides[slideIDX]
           self.init_read_class(self.size, self.tdatadict[slideIDX])
       if self.images_index[slideIDX] >= len(self.images[slideIDX]):
           flag, self.images[slideIDX] = self.read_class[self.recent_slide].getImage()
           self.images_index[slideIDX] = 0
           if not flag:
               return None, None
       # print('img_id', self.images_index[slideIDX], 'img_len', len(self.images[slideIDX]))
       img = self.images[slideIDX][self.images_index[slideIDX]].array
       x = self.images[slideIDX][self.images_index[slideIDX]].start_x
       y = self.images[slideIDX][self.images_index[slideIDX]].start_y
       target = self.xy2target[slideIDX]['%d,%d' % (int(x), int(y))]
       self.targets += [target]
       self.images_index[slideIDX] += 1
       if self.visualized <= 9 and random.randint(0,9) == 0:
           cv2.imwrite(os.path.join(self.output, '%s_ID%d_label%d.jpg'%(self.name,self.visualized,target)), img)
           self.visualized += 1
       if self.transform is not None:
           img = self.transform(img)
       return img, target
   def __len__(self):
       return len(self.t_data)
   def collect_fn(self, batch):
       new_b = [b for b in batch if b[0] is not None]
       return torch.stack([b[0] for b in new_b]), torch.stack([torch.as_tensor(b[1]) for b in new_b])

if __name__ == '__main__':
    weights_path = "D:/WSI_analysis/cls/sRMBv02-cls/yolov3_ckpt_step_1649998.pth"
    mode = 'cls'
    model = Darknet(model_def, img_size=img_size, lite_mode=True, use_final_loss=True, debug_mode=False, old_version=False, mode=mode).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    accs1, tprs1, tnrs1, auc1 = main(model, inference_icn)
    weights_path = "D:/WSI_analysis/cls/sRMBv02/yolov3_custom_best.pth"
    mode = 'det'
    model = Darknet(model_def, img_size=img_size, lite_mode=True, use_final_loss=True, debug_mode=False, old_version=False, mode=mode).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    accs2, tprs2, tnrs2, auc2 = main(model, inference_icn)

    weights_path = "D:/WSI_analysis/cls/mobilenetv2_pretrained/checkpoint_best.pth"
    model = models.mobilenet.MobileNetV2(num_classes=2).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device)['state_dict'])
    accs3, tprs3, tnrs3, auc3 = main(model, inference_mobilenet, is_norm=True)

    print('YOCO')
    print('Acc1', np.mean(accs1),np.std(accs1))
    print('TPR1', np.mean(tprs1),np.std(tprs1))
    print('TNR1', np.mean(tnrs1),np.std(tnrs1))
    print('AUC1', np.mean(auc1),np.std(auc1))
    print('YOLCO')
    print('Acc2', np.mean(accs2),np.std(accs2))
    print('TPR2', np.mean(tprs2),np.std(tprs2))
    print('TNR2', np.mean(tnrs2),np.std(tnrs2))
    print('AUC2', np.mean(auc2),np.std(auc2))
    print('MNV2')
    print('Acc3', np.mean(accs3),np.std(accs3))
    print('TPR3', np.mean(tprs3),np.std(tprs3))
    print('TNR3', np.mean(tnrs3),np.std(tnrs3))
    print('AUC3', np.mean(auc3),np.std(auc3))
    input("Press any key to exit")
    exit()
