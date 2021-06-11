import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from datetime import datetime
import random
import MySQLdb
import matplotlib.pyplot as plt
# import openslide
import numpy as np

Anchors = {
    'stage4': [[102, 101], [178, 176], [263, 261]],
    'stage5': [[369, 363], [524, 520], [822, 829]]
}


def str2slide(file):
    slide = ''
    for s in file.split('.')[:-1]:
        slide += s + '.'
    slide = slide[:-1]
    return slide
    
def prepareLabels(label_lines, balance=True):
    pos_label = ['Positive', 'pos', '_p', 'sfy9']
    neg_label = ['Negative', 'neg']
    slideLable = {}
    posn = 0
    negn = 0
    posSlides = []
    negSlides = []
    for line in label_lines:
        file = line.split('\\')[-1].split('/')[-1]
        slide = str2slide(file)
        slideLable[slide] = None
        for p in pos_label:
            if p in line:
                slideLable[slide] = 1
                posn += 1
                posSlides.append(slide)
                break
        for n in neg_label:
            if n in line:
                slideLable[slide] = 0
                negn += 1
                negSlides.append(slide)
                break

    return slideLable, posn, negn, posSlides, negSlides


def distanceConserve(x, y, xlist, ylist, thres=1000):
    for xi, yi in zip(xlist, ylist):
        if ((x - xi) ** 2 + (y - yi) ** 2) ** 0.5 <= thres:
            return False
    return True


class ImgDataset(Dataset):
    def __init__(self,
                 slide_list_txt,
                 yolo_root,
                 name='train',
                 top_n=100,
                 bottom_n=100,
                 balance=True,
                 joint=0,
                 data_root='sfy1&2',):
        self.yolo_root = yolo_root
        self.name = name
        self.balance = balance
        self.top_n = top_n
        self.bottom_n = bottom_n
        with open(slide_list_txt, 'r') as f:
            label_lines = f.read().split('\n')[:-1]
        self.slide2label, self.posn, self.negn, self.posSlides, self.negSlides = prepareLabels(label_lines, balance)
        if self.balance:
            sliden = min(self.posn, self.negn)
            self.slides = random.sample(self.posSlides, sliden) + random.sample(self.negSlides, sliden)
        else:
            self.slides = self.posSlides + self.negSlides
        self.db = MySQLdb.connect("192.168.0.160", 'weiziquan','123','das',charset='utf8')

    def __getitem__(self, i):
        slide = self.slides[i]
        label = self.slide2label[slide]
        cursor = self.db.cursor()
        cursor.execute('SELECT slide_path, slide_name FROM das.slides where slide_name LIKE "{}%"'.format(slide))
        results = cursor.fetchall()
        if len(results) == 0:
            print('DB Error')
            exit()
        for slide_path, slide_name in results:
            if os.path.exists(os.path.join(slide_path, slide_name)) and 'srp' not in slide_name:
                break
        if not os.path.exists(os.path.join(r'D:\WSI_analysis\rnn_input\%s_sequences%s' % (self.data_root, self.name), slide + '.pth')):
            print('Need prepare data')
            exit()
        data = torch.load(os.path.join(r'D:\WSI_analysis\rnn_input\%s_sequences%s' % (self.data_root, self.name), slide+'.pth'))['data']
        data = torch.cat([data[-1 * self.top_n:], data[:self.bottom_n]])
        # slide_handle = openslide.OpenSlide(os.path.join(slide_path, slide_name))
        # for seq in data:
        #     for
        #     x, y, w, h = seq[]
        #     region = np.array(slide_handle.read_region((int(x), int(y)), self.level, (int(w), int(h))))

        return {'info': slide, 'data': data, 'label': label}


class MaxpoolDataset(Dataset):
    def __init__(self,
                 fm_root,
                 slide_list_txt,
                 yolo_root,
                 transform=None,
                 keys=None,
                 name='train',
                 top_n=100,
                 bottom_n=100,
                 balance=True,
                 dis_thers=None,
                 joint=0,
                 data_root='sfy1&2',
                 ):
        self.maxpooler = torch.nn.MaxPool2d(stride=top_n, kernel_size=bottom_n)
        self.name = name
        self.fm_root = fm_root
        self.balance = balance
        if keys is None:
            keys = ['stage4', 'stage5']
        self.keys = keys
        self.transform = transform
        with open(slide_list_txt, 'r') as f:
            label_lines = f.read().split('\n')[:-1]
        self.slide2label, self.posn, self.negn, self.posSlides, self.negSlides = prepareLabels(label_lines, balance)
        if self.balance:
            sliden = min(self.posn, self.negn)
            self.slides = random.sample(self.posSlides, sliden) + random.sample(self.negSlides, sliden)
        else:
            self.slides = self.posSlides + self.negSlides

    def setSlides(self):
        if self.balance:
            sliden = min(self.posn, self.negn)
            self.slides = random.sample(self.posSlides, sliden) + random.sample(self.negSlides, sliden)

    def __getitem__(self, i):
        slide = self.slides[i]
        label = self.slide2label[slide]
        if not os.path.exists(os.path.join(r'D:\WSI_analysis\rnn_input\%s_maxpooled%s' % (self.data_root, self.name), slide + '.pth')):
            os.makedirs(r'D:\WSI_analysis\rnn_input\%s_maxpooled%s' % (self.data_root, self.name), exist_ok=True)
            fm = torch.load(os.path.join(self.fm_root, slide + '.pth'))['fm']
            data = self.maxpooler(fm).reshape(-1, fm.shape[0])
            torch.save({'data': data}, os.path.join(r'D:\WSI_analysis\rnn_input\%s_maxpooled%s' % (self.data_root, self.name), slide+'.pth'))
        else:
            data = torch.load(os.path.join(r'D:\WSI_analysis\rnn_input\%s_maxpooled%s' % (self.data_root, self.name), slide+'.pth'))['data']
        return {'info': slide, 'data': data, 'label': label}

    def __len__(self):
        return len(self.slides)


class SequenceDataset(Dataset):
    def __init__(self,
                 fm_root,
                 slide_list_txt,
                 yolo_root,
                 transform=None,
                 keys=None,
                 one_term_len=100,
                 anchor_num=3,
                 name='train',
                 top_n=100,
                 bottom_n=100,
                 balance=True,
                 dis_thers=10,
                 joint=0,
                 data_root='sfy1&2',
                 ):
        # if top_n > one_term_len or bottom_n > one_term_len:
        #     print("Error Arguments")
        #     exit()
        self.data_root = data_root
        self.name = name
        self.balance = balance
        self.top_n = top_n
        self.bottom_n = bottom_n
        self.maxs = {}
        self.mins = {}
        if keys is None:
            keys = ['stage4', 'stage5']
        self.keys = keys
        self.one_term_len = int(one_term_len / len(keys))
        self.anchor_num = anchor_num
        self.transform = transform
        self.joint = joint
        with open(slide_list_txt, 'r') as f:
            label_lines = f.read().split('\n')[:-1]
        self.slide2label, self.posn, self.negn, self.posSlides, self.negSlides = prepareLabels(label_lines, balance)
        if self.balance:
            sliden = min(self.posn, self.negn)
            self.slides = random.sample(self.posSlides, sliden) + random.sample(self.negSlides, sliden)
        else:
            self.slides = self.posSlides + self.negSlides
        self.fm_exists = os.path.exists(os.path.join(fm_root, self.posSlides[0]+'.pth'))
        self.withWH = True if 'withWH' in name else False
        if self.withWH:
            temp = torch.load(r'D:\WSI_analysis\mechine_learning\v4_withWH_temp.pth')
            self.mins = temp['mins']
            self.maxs = temp['maxs']
        if self.fm_exists:
            self.fm_root = fm_root
            self.yolo_root = yolo_root
            fm = torch.load(os.path.join(self.fm_root, self.posSlides[0]+'.pth'))['fm']
            xs = torch.zeros(anchor_num, fm.shape[-2], fm.shape[-1])
            ys = torch.zeros(anchor_num, fm.shape[-2], fm.shape[-1])
            for i in range(fm.shape[-1]):
                ys[:, :, i] = i
            for i in range(fm.shape[-2]):
                xs[:, i, :] = i
            self.xs = xs.reshape(-1)
            self.ys = ys.reshape(-1)
            self.index = torch.IntTensor([i for i in range(len(self.xs))]).reshape(anchor_num, fm.shape[-2], fm.shape[-1])
            self.anchor_w = torch.zeros(anchor_num, fm.shape[-2], fm.shape[-1])
            self.anchor_h = torch.zeros(anchor_num, fm.shape[-2], fm.shape[-1])
            self.prepareTemp(fm.shape[-1], name, dis_thers, withWH=self.withWH)

        print('################################################')
        print('# Dataset prepared! Slide len:', len(self.slides))
        print('# Dataset consists of ', self.posn, 'pos,', self.negn, 'neg')
        print('################################################')

    def __getitem__(self, i):

        slide = self.slides[i]
        label = self.slide2label[slide]
        __root = r'D:\WSI_analysis\rnn_input\%s_sequences%s' % (self.data_root, self.name)
        __root0 = '/mnt/160_d/WSI_analysis/rnn_input/%s_sequences%s' % (self.data_root, self.name)
        if not os.path.exists(os.path.join(__root, slide + '.pth')) and not os.path.exists(os.path.join(__root0, slide + '.pth')):
            if not self.fm_exists:
                print('%s does not exist! Need feature map' % os.path.join(r'D:\WSI_analysis\rnn_input\%s_sequences%s' % (self.data_root, self.name), slide + '.pth'))
                exit()
            os.makedirs(r'D:\WSI_analysis\rnn_input\%s_sequences%s' % (self.data_root, self.name), exist_ok=True)
            if 'yolo' not in self.name:
                fm = torch.load(os.path.join(self.fm_root, slide+'.pth'))['fm']
            else:
                x = torch.load(os.path.join(self.yolo_root, slide + '.pth'))
                fm = torch.cat([x['stage5'][:, 4:-4, 4:-4], torch.max_pool2d(x['stage4'][:, 4:-4, 4:-4], 2, 2)])
            data = self.prepareSequence(fm, self.mins[slide], self.maxs[slide])
            torch.save({'data': data}, os.path.join(r'D:\WSI_analysis\rnn_input\%s_sequences%s' % (self.data_root, self.name), slide+'.pth'))
        else:
            if not os.path.exists(__root):
                __root = __root0
            if not os.path.exists(__root):
                print('Error in', __root)
                exit()
            data = torch.load(os.path.join(__root, slide+'.pth'))
            try:
                outputs = data['outputs']
            except:
                outputs = None
            try:
                mins = data['mins']
                maxs = data['maxs']
            except:
                mins, maxs = None, None
            try:
                data = data['data']
            except:
                data = data['datayx']
        if not self.fm_exists:
            lenmaxs = int(len(data) / 2)
            lenmins = int(len(data) / 2)
        else:
            lenmaxs = len(self.maxs[slide])
            lenmins = len(self.mins[slide])
        if self.top_n == -1:
            top_n = lenmaxs
        elif self.top_n == 0:
            top_n = -1 * len(data)
        else:
            top_n = self.top_n if self.top_n <= lenmaxs else lenmaxs
        if self.bottom_n == -1:
            bottom_n = lenmins
        else:
            bottom_n = self.bottom_n if self.bottom_n <= lenmins else lenmins
        if self.withWH and self.top_n != -1 and self.bottom_n != -1:
            # if not self.fm_exists:
            #     print('%s does not exist! Need feature map' % os.path.join(r'D:\WSI_analysis\rnn_input\%s_sequences%s' % (self.data_root, self.name), slide + '.pth'))
            #     exit()
            lenmaxs = len(self.maxs[slide])
            lenmins = len(self.mins[slide])
            data = self.average_sample(slide, data, top_n, bottom_n, lenmaxs, lenmins)
        else:
            data = self.top_sample(data, top_n, bottom_n)
        # _ind = [index for index in range(len(data))]
        # random.shuffle(_ind)
        # data = data[_ind]
        return {'info': slide, 'data': data, 'label': label, 'mins': mins, 'maxs': maxs, 'outputs': outputs}

    def average_sample(self, slide, data, top_n, bottom_n, lenmaxs, lenmins):
        top_step = int(lenmaxs / top_n) + 1
        bottom_step = int(lenmins / bottom_n) + 1 if bottom_n != 0 else None
        if top_n < 0:
            top_data = data[-1*top_n:]
        else:
            top_data = data[-1*lenmaxs::top_step]
        if bottom_n == 0:
            bottom_data = data[:bottom_n]
        else:
            bottom_data = data[:lenmins:bottom_step]
        data = torch.cat([top_data, bottom_data])
        if len(data) < self.top_n + self.bottom_n:
            data = torch.cat([data, torch.zeros(self.top_n + self.bottom_n - len(data), data.shape[1])])
        return data

    def top_sample(self, data, top_n, bottom_n):
        top_port = 2 * self.one_term_len * len(self.keys) / top_n if top_n != 0 else 1.1
        bottom_port = 2 * self.one_term_len * len(self.keys) / bottom_n if bottom_n != 0 else 1.1
        if int(top_port) == top_port and len(data) == 2 * 2 * self.one_term_len * len(self.keys):
            tops = [torch.from_numpy(data.numpy()[::-1][i*2*self.one_term_len:(i+1)*2*self.one_term_len][:int(top_n/len(self.keys))].copy()) for i in range(len(self.keys))]
        else:
            tops = [data[-1 * top_n:]]
        if int(bottom_port) == bottom_port and len(data) == 2 * 2 * self.one_term_len * len(self.keys):
            bottoms = [data[i*2*self.one_term_len:(i+1)*2*self.one_term_len][:int(bottom_n/len(self.keys))] for i in range(len(self.keys))]
        else:
            bottoms = [data[:bottom_n]]
        data = torch.cat(tops + bottoms)
        if len(data) < self.top_n + self.bottom_n:
            data = torch.cat([data, torch.zeros(self.top_n + self.bottom_n - len(data), data.shape[1])])
        return data

    def __len__(self):
        return len(self.slides)

    def setSlides(self):
        if self.balance:
            sliden = min(self.posn, self.negn)
            self.slides = random.sample(self.posSlides, sliden) + random.sample(self.negSlides, sliden)

    def prepareTemp(self, _shape, name, disThres, withWH=False):
        if os.path.exists('v4_%stemp.pth' % name):
            temp = torch.load('v4_%stemp.pth' % name)
            self.mins = temp['mins']
            self.maxs = temp['maxs']
            flag = False
            for slide in self.slides:
                if slide not in self.mins or slide not in self.maxs:
                    self.mins[slide], self.maxs[slide] = self.getMinsMaxs(slide, _shape, disThres=disThres, withWH=withWH)
                    flag = True
            if flag:
                torch.save({'mins': self.mins, 'maxs': self.maxs}, 'v4_%stemp.pth' % name)
            return

        for slide in tqdm(self.slides, desc='Preparing Tmp Data'):
            self.mins[slide], self.maxs[slide] = self.getMinsMaxs(slide, _shape, disThres=disThres, withWH=withWH)
        torch.save({'mins': self.mins, 'maxs': self.maxs}, 'v4_%stemp.pth' % name)

    def prepareSequence(self, fm, mins, maxs):
        data = torch.zeros(len(mins)+len(maxs), fm.shape[0])
        data_index = 0
        for xi, yi in zip(self.xs[mins], self.ys[mins]):
            data[data_index] = fm[:, int(xi.item()), int(yi.item())]
            data_index += 1
        for xi, yi in zip(self.xs[maxs], self.ys[maxs]):
            data[data_index] = fm[:, int(xi.item()), int(yi.item())]
            data_index += 1
        return data

    def getMinsMaxs(self, slide, _shape, disThres=10, withWH=False):
        yolos = torch.load(os.path.join(self.yolo_root, slide + '.pth'))
        _mins = []
        _maxs = []
        for k in self.keys:
            if yolos[k].shape[-1] != _shape:
                yolos[k] = yolos[k][:, 4:-4, 4:-4]
            if yolos[k].shape[-1] != _shape:
                yolos[k] = torch.max_pool2d(yolos[k], 2, 2)
            yolos[k] = yolos[k].view(self.anchor_num, 7, yolos[k].shape[1], yolos[k].shape[2]).permute(0, 2, 3,
                                                                                                       1).contiguous()
            if withWH:
                x = torch.sigmoid(yolos[k][..., 0])  # x
                y = torch.sigmoid(yolos[k][..., 1])  # y
                w = yolos[k][..., 2]  # x
                h = yolos[k][..., 3]  # y
                anchors = Anchors[k]
                self.anchor_w = torch.zeros_like(w)
                self.anchor_h = torch.zeros_like(h)
                for i in range(self.anchor_num):
                    self.anchor_w[i, :, :] = anchors[i][0]
                    self.anchor_h[i, :, :] = anchors[i][1]
                self.anchor_w = self.anchor_w.reshape(-1)
                self.anchor_h = self.anchor_h.reshape(-1)
            else:
                x, y, w, h = None, None, None, None
            conf = torch.sigmoid(yolos[k][..., 4] * torch.exp(-1 * yolos[k][..., 5]))  # conf
            fp_conf = torch.sigmoid(yolos[k][..., 5])  # fp_conf
            mins, maxs = self.getMaxsMinsFromYolo(conf, disThres, x, y, w, h)
            fpmins, fpmaxs = self.getMaxsMinsFromYolo(fp_conf, disThres, x, y, w, h)
            _mins = _mins + [mins] + [fpmins]
            _maxs = _maxs + [fpmaxs] + [maxs]

        _mins = torch.cat(_mins)
        _maxs = torch.cat(_maxs)
        return _mins, _maxs

    def getMaxsMinsFromYolo(self, yolo_map, disThres, x=None, y=None, w=None, h=None):
        maxargsort = torch.argsort(yolo_map.reshape(-1))
        minargsort = torch.argsort(-1.0 * yolo_map.reshape(-1))
        if disThres == 0 and x is None:
            maxs = maxargsort[:self.one_term_len]
            mins = minargsort[:self.one_term_len]
        elif x is not None and y is not None and w is not None and h is not None:
            maxs, mins = self.buildMinsMaxsWithWH(maxargsort, minargsort, x.reshape(-1), y.reshape(-1), w.reshape(-1), h.reshape(-1))
        else:
            mins, maxs = self.buildMinsMaxsWithThres(maxargsort, minargsort, disThres)
        return mins, maxs

    def buildMinsMaxsWithThres(self, maxargsort, minargsort, disThres):
        maxs, mins = [], []
        i = 0
        while len(maxs) < self.one_term_len:
            x, y = self.xs[maxargsort[i]], self.ys[maxargsort[i]]
            if len(maxs) > 1:
                xlist, ylist = self.xs[torch.stack(maxs)].tolist(), self.ys[torch.stack(maxs)].tolist()
            else:
                xlist, ylist = self.xs[maxs].tolist(), self.ys[maxs].tolist()
                if type(xlist) is float or type(ylist) is float:
                    xlist, ylist = [xlist], [ylist]
            if distanceConserve(x, y, xlist, ylist, thres=disThres):
                maxs += [maxargsort[i]]
            i += 1
        maxs = torch.stack(maxs)
        i = 0
        while len(mins) < self.one_term_len:
            x, y = self.xs[minargsort[i]], self.ys[minargsort[i]]
            if len(mins) > 1:
                xlist, ylist = self.xs[torch.stack(mins)].tolist(), self.ys[torch.stack(mins)].tolist()
            else:
                xlist, ylist = self.xs[mins].tolist(), self.ys[mins].tolist()
                if type(xlist) is float or type(ylist) is float:
                    xlist, ylist = [xlist], [ylist]
            if distanceConserve(x, y, xlist, ylist, thres=disThres):
                mins += [minargsort[i]]
            i += 1
        mins = torch.stack(mins)
        return mins, maxs

    def buildMinsMaxsWithWH(self, maxargsort, minargsort, x, y, w, h, scale=32):
        maxs, mins = [], []
        select_factor = 1/4
        select_flag = torch.zeros_like(self.index)
        for i in range(self.one_term_len):
            _x, _y = (self.xs[maxargsort[i]] + x[maxargsort[i]])*scale, (self.ys[maxargsort[i]] + y[maxargsort[i]])*scale
            _w, _h = w[maxargsort[i]]*self.anchor_w[maxargsort[i]]*scale*select_factor, h[maxargsort[i]]*self.anchor_h[maxargsort[i]]*scale*select_factor
            start_x, end_x = int((_x - _w/2)/scale), int((_x + _w/2)/scale)
            start_y, end_y = int((_y - _h/2)/scale), int((_y + _h/2)/scale)
            maxs += [maxargsort[i]]
            for xi in range(0 if start_x < 0 else start_x,
                            self.index.shape[-2] - 1 if end_x >= self.index.shape[-2] else end_x):
                for yi in range(0 if start_y < 0 else start_y,
                                self.index.shape[-1] - 1 if end_y >= self.index.shape[-1] else end_y):
                    if select_flag[0, xi, yi] == 0:
                        maxs += [self.index[0, xi, yi]]
                        select_flag[0, xi, yi] = 1
        select_flag = torch.zeros_like(self.index)
        for i in range(self.one_term_len):
            _x, _y = (self.xs[minargsort[i]] + x[minargsort[i]])*scale, (self.ys[minargsort[i]] + y[minargsort[i]])*scale
            _w, _h = w[minargsort[i]]*self.anchor_w[minargsort[i]]*scale*select_factor, h[minargsort[i]]*self.anchor_h[minargsort[i]]*scale*select_factor
            start_x, end_x = int((_x - _w/2)/scale), int((_x + _w/2)/scale)
            start_y, end_y = int((_y - _h/2)/scale), int((_y + _h/2)/scale)
            mins += [minargsort[i]]
            for xi in range(0 if start_x < 0 else start_x,
                            self.index.shape[-2] - 1 if end_x >= self.index.shape[-2] else end_x):
                for yi in range(0 if start_y < 0 else start_y,
                                self.index.shape[-1] - 1 if end_y >= self.index.shape[-1] else end_y):
                    if select_flag[0, xi, yi] == 0:
                        mins += [self.index[0, xi, yi]]
                        select_flag[0, xi, yi] = 1
        maxs = torch.stack(maxs)
        mins = torch.stack(mins)
        return mins, maxs

