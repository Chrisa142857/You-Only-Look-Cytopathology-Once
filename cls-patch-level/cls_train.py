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

from tqdm import tqdm
from slide_readtool import fast_read
from cls_icn import MobileNetICN
_ROOT = r'D:\WSI_analysis\cls'

parser = argparse.ArgumentParser(description='MIL-nature-medicine-2019 tile classifier training script')
parser.add_argument('--train_lib', type=str, default=r'%s\data\det-train.pth'%_ROOT, help='path to train MIL library binary')
parser.add_argument('--val_lib', type=str, default=r'%s\data\det-valid.pth'%_ROOT, help='path to validation MIL library binary. If present.')
parser.add_argument('--output', type=str, default='mobilenetICN_pretrained', help='name of output file')
parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size (default: 512)')
parser.add_argument('--nepochs', type=int, default=1000, help='number of epochs')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--test_every', default=10, type=int, help='test on val every (default: 10)')
parser.add_argument('--weights', default=0.5, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
parser.add_argument('--gpu', default='1', type=str)
parser.add_argument('--resume', default='none', type=str, help='last | best | none')

best_acc = 0
def main():
    global args, best_acc
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.output, exist_ok=True)
    #cnn
    model = MobileNetICN(num_classes=2)
    # model = models.mobilenet_v2(pretrained=True, num_classes=2)
    if args.resume == "last" or args.resume == 'best':
      ch=torch.load(os.path.join(args.output,'checkpoint_%s.pth'%args.resume))
      model.load_state_dict(ch['state_dict'])
      start_epoch=ch['epoch'] - 1
    else:
      start_epoch=0
    params_n = 0
    for l in model.features:
        for p in l.parameters():
            params_n += p.reshape(-1).shape[0]
    print(model)
    print("params_n: ", params_n)
    model.cuda()

    if args.weights==0.5:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        w = torch.Tensor([1-args.weights,args.weights])
        criterion = nn.CrossEntropyLoss(w).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    if args.resume == "last" or args.resume == 'best':
      optimizer.load_state_dict(ch['optimizer'])

    cudnn.benchmark = True

    #normalization
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    #load data
    train_dset = CLSdataset(libraryfile=args.train_lib, transform=trans, name='train')
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, collate_fn=train_dset.collect_fn)
    if args.val_lib:
        val_dset = CLSdataset(libraryfile=args.val_lib, transform=trans, name='valid')
        val_loader = torch.utils.data.DataLoader(
            val_dset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False, collate_fn=val_dset.collect_fn)

    #open output file
    fconv = open(os.path.join(args.output,'convergence.csv'), 'w')
    fconv.write('epoch,metric,value\n')
    fconv.close()
    #loop throuh epochs
    for epoch in range(start_epoch, args.nepochs):
        #Validation
        if args.val_lib and (epoch+1) % args.test_every == 0:
            val_dset.shuffledata()
            probs, targets = inference(epoch, val_loader, model)
            pred = [1 if x >= 0.5 else 0 for x in probs]
            err,fpr,fnr = calc_err(pred, targets)
            print('Validation\tEpoch: [{}/{}]\tError: {}\tFPR: {}\tFNR: {}'.format(epoch+1, args.nepochs, err, fpr, fnr))
            fconv = open(os.path.join(args.output, 'convergence.csv'), 'a')
            fconv.write('{},error,{}\n'.format(epoch+1, err))
            fconv.write('{},fpr,{}\n'.format(epoch+1, fpr))
            fconv.write('{},fnr,{}\n'.format(epoch+1, fnr))
            fconv.close()
            #Save best model
            err = (fpr+fnr)/2.
            if 1-err >= best_acc:
                best_acc = 1-err
                obj = {
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict()
                }
                torch.save(obj, os.path.join(args.output,'checkpoint_best.pth'))

        train_dset.shuffledata()
        loss = train(epoch, train_loader, model, criterion, optimizer)
        print('Training\tEpoch: [{}/{}]\tLoss: {}'.format(epoch + 1, args.nepochs, loss))
        fconv = open(os.path.join(args.output, 'convergence.csv'), 'a')
        fconv.write('{},loss,{}\n'.format(epoch + 1, loss))
        fconv.close()
        obj = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict()
        }
        torch.save(obj, os.path.join(args.output, 'checkpoint_last.pth'))


def inference(run, loader, model):
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

def train(run, loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.
    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*input.size(0)
    return running_loss/len(loader.dataset) if len(loader.dataset)!=0 else -1

def calc_err(pred,real):
    pred = np.array(pred)
    real = np.array(real)
    neq = np.not_equal(pred, real)
    err = float(neq.sum())/pred.shape[0]
    fpr = float(np.logical_and(pred==1,neq).sum())/(real==0).sum()
    fnr = float(np.logical_and(pred==0,neq).sum())/(real==1).sum()
    return err, fpr, fnr

def random_sample_xy(xmin, ymin, xmax, ymax, size):
    bw, bh = xmax-xmin, ymax-ymin
    if xmin > xmax-size:
      x = (xmax + xmin - size) / 2
    elif xmin+int(0.25*bw) > xmax-int(0.25*bw)-size: 
      x = random.randint(xmin, xmax-size)
    else:
      x = random.randint(xmin+int(0.25*bw), xmax-int(0.25*bw)-size)
    if ymin > ymax-size:
      y = (ymax + ymin - size) / 2
    elif ymin+int(0.25*bh) > ymax-int(0.25*bh)-size: 
      y = random.randint(ymin, ymax-size)
    else:
      y = random.randint(ymin+int(0.25*bh), ymax-int(0.25*bh)-size)
    return x, y

def prepare_tdata(slide2IDX, partial_ann, size, roi):
    tdata = {}
    num = 0
    if size >= roi: 
      print('Error: ROI setting {} < Patch-Size {}'.format(roi, size))
      exit()
    for s in partial_ann:
      if s is None: continue
      tdata[slide2IDX[s]] = []
      hitted = []
      for xmin, ymin, xmax, ymax in partial_ann[s]:
        if len([0 for x1, y1, x2, y2 in hitted if xmin==x1 and ymin==y1 and xmax==x2 and ymax==y2]) > 0: continue
        hitted += [[xmin, ymin, xmax, ymax]]
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        x, y = random_sample_xy(xmin, ymin, xmax, ymax, size)
        flag = True
        while flag:
          nx, ny = random.randint(x+size/2-roi/2, x+size/2+roi/2), random.randint(y+size/2-roi/2, y+size/2+roi/2)
          flag = len([0 for x1, y1, x2, y2 in partial_ann[s] if (x1<nx<x2 and y1<ny<y2) or (x1<nx+size<x2 and y1<ny+size<y2)]) > 0
        tdata[slide2IDX[s]] += [[x, y, 1], [nx, ny, 0]]
      num += len(tdata[slide2IDX[s]])
    print('Number of tiles:', num)
    torch.save(tdata, 'temp_xy.pth')
    return tdata

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
    main()
