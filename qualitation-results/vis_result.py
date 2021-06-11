import matplotlib.pyplot as plt
import torch
import os
from dataset import str2slide, prepareLabels, SequenceDataset
from det_model import Darknet
# from cls_model import create_model
import datetime
import numpy as np
from slide_readtool import fast_read
from utils.pysrp.pysrp import Srp
import random
import cv2

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    tf = max(tl - 1, 1)  # font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)        
    cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, color, thickness=tf, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        # cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        # cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        zeros_mask = np.zeros((img.shape), dtype=np.uint8)
        cv2.rectangle(zeros_mask, c1, c2, color, -1, cv2.LINE_AA)  # filled
        img=cv2.addWeighted(img,1,zeros_mask,1,0)
    return img

def get_params(name, model):
	if 'transformer' in model:
		n_layers = 10
	else:
		n_layers = 1
	if 'yolo' not in name:
	    input_dim = 768
	elif 'yolotiny' in name:
	    input_dim = 768
	else:
	    if 'yolov3' in name and 'New' in name:
	        input_dim = 1792
	    elif 'yolov3' in name:
	        input_dim = 1536
	    else:
	        input_dim = 42
	return input_dim, n_layers, 2048


def get_xy(mins, maxs, side=1400):
	xs = torch.zeros(anchor_num, side, side)
	ys = torch.zeros(anchor_num, side, side)
	for i in range(side):
	    ys[:, :, i] = i
	for i in range(side):
	    xs[:, i, :] = i
	xs = xs.reshape(-1)
	ys = ys.reshape(-1)
	return torch.stack([xs[mins], ys[mins]], dim=1), torch.stack([xs[maxs], ys[maxs]], dim=1)

def sort_output(yolo_map, one_term_len=50):
	maxargsort = torch.argsort(yolo_map.reshape(-1))
	minargsort = torch.argsort(-1.0 * yolo_map.reshape(-1))
	maxs = maxargsort[:one_term_len]
	mins = minargsort[:one_term_len]
	return maxs, mins


def compute_grid_offsets(grid_size, img_dim, anchors):
	num_anchors = len(anchors)
	g_x = grid_size[1]
	g_y = grid_size[0]
	stride_w = img_dim[1] / g_x
	stride_h = img_dim[0] / g_y
	# Calculate offsets for each grid
	grid_x = torch.arange(g_x).repeat(g_y, 1).view([1, 1, g_y, g_x])
	grid_y = torch.arange(g_y).repeat(g_x, 1).t().view([1, 1, g_y, g_x])
	scaled_anchors = torch.FloatTensor([(a_w / stride_w, a_h / stride_h) for a_w, a_h in anchors])
	anchor_w = scaled_anchors[:, 0:1].view((1, num_anchors, 1, 1))
	anchor_h = scaled_anchors[:, 1:2].view((1, num_anchors, 1, 1))
	return grid_x, grid_y, anchor_w, anchor_h, stride_w, stride_h

Anchors = {
    'icn': [[[369, 363], [524, 520], [822, 829]], [[102, 101], [178, 176], [263, 261]]],
    'yolov3': [[[135,169], [344,319]], [[37,58], [81,82]], [[10,14], [23,27]]],  
    'yolotiny': [[[369, 363], [524, 520], [822, 829]], [[102, 101], [178, 176], [263, 261]]]
    # 'yolotiny': [[[81,82],  [135,169],  [344,319]], [[23,27],  [37,58],  [81,82]]]
}

inds = [66, 1462, 1354, 1834, 713, 879, 448, 866, 810, 585]
labels = ['s1-p', 's2-n', 's2-n', 's4-n', 's3-p', 's4-p', 's3-p', 's4-p', 's4-p', 's3-p']
nn, pn=3, 7
s1n, s2n, s3n, s4n=1, 2, 3, 4
slides = [
	'O:/WSIData/TransSRP/sfy1/3DHisTech/Positive/16032022 yulan Ascus 4.srp',
	'O:/WSIData/TransSRP/sfy2/3DHisTech/Negative/1156336 0893088.srp',
	'O:/WSIData/TransSRP/sfy2/3DHisTech/Negative/1156802 0893014.srp',
	'O:/WSIData/TransSRP/sfy3/WNLO/negative/L230/2018-11-12-163303-842.srp',
	'O:/WSIData/TransSRP/sfy9/SZSQ/sfy1110079 0893001.srp',
	'O:/WSIData/TransSRP/sfy4/WNLO/Shengfuyou_4th_p/1615547 2226196.srp',
	'O:/WSIData/TransSRP/sfy3/SZSQ/positive/Shengfuyou_3th_positive_40X/1169519 0893136.srp',
	'O:/WSIData/TransSRP/sfy4/WNLO/Shengfuyou_4th_p/1615259 2226231.srp',
	'O:/WSIData/TransSRP/sfy3/WNLO/positive/L250/2018-11-01-172343-637.srp',
	'O:/WSIData/TransSRP/sfy7/SZSQ/shengqiang_40x/Shengfuyou_7th_positive_40x/1153063.srp'
]
slide2path = {}
slide2ind = {}
whlist, levellist = {}, {}
for i, srp_file in enumerate(slides):
	slide2path[srp_file.split('/')[-1].replace('.srp', '')] = srp_file
	slide2ind[srp_file.split('/')[-1].replace('.srp', '')] = i
	srp_handle = Srp()
	srp_handle.open(srp_file)
	h, w = srp_handle.getAttrs()['width'], srp_handle.getAttrs()['height']
	if '40x' in srp_file or '40X' in srp_file:
	    whlist[srp_file.split('/')[-1].replace('.srp', '')] = [int(w/2), int(h/2)]
	    levellist[srp_file.split('/')[-1].replace('.srp', '')] = 1
	else:
	    whlist[srp_file.split('/')[-1].replace('.srp', '')] = [w, h]
	    levellist[srp_file.split('/')[-1].replace('.srp', '')] = 0

for k in whlist:
	swh = whlist[k]
	print(slide2ind[k])
	print(int(swh[1]/(2**8)), int(swh[0]/(2**8)))
exit()
# =====================================================================================================================
import matplotlib as mpl
def contraction_wsi(startx, starty, swh, bx_max, by_max, conf_max, read_class, label, marker, title, fig, ax, show_img, level):
	print('bx_max.shape', bx_max.shape)
	ol = 8
	level += ol
	# scale = 256 / int(swh[0]/(2**ol))
	# ax.set_autoscale_on(False)
	if show_img:
		wsi = read_class.getTile(level, 0, 0, int(swh[1]/(2**ol)), int(swh[0]/(2**ol))).array
		print('wsi.shape', wsi.shape)
		# wsi = cv2.resize(wsi, (int((swh[1]/(2**ol))*scale), 256))
		ax.imshow(wsi[:,:,::-1])
	xs, ys, cs = [], [], []
	for x, y, c in zip(bx_max, by_max, conf_max):
		flag = True
		for ux, uy in zip(xs, ys): flag = abs(x-ux)>=512 or abs(y-uy)>=512
		if flag: 
			xs.append(x)
			ys.append(y)
			cs.append(c)
	ax.scatter([int((startx + x)/(2**ol)) for x in xs], [int((starty + y)/(2**ol)) for y in ys], label=None, c=cs, cmap='rainbow', marker=marker, s=15, linewidths=0, alpha=0.5)
	ax.axis('off')
	# if show_img:
	# 	ax.set_ylim([wsi.shape[0]-500, wsi.shape[0]])
	ax.set_title(title)


def save_bbox(startx, starty, bx_max, by_max, bw_max, bh_max, conf_max, read_class, save_root, slide, label, tile_side, color, level, j):
	for tilei in range(len(bx_max)):
		x, y, w, h, c = bx_max[tilei], by_max[tilei], bw_max[tilei], bh_max[tilei], conf_max[tilei]
		x = int(startx + x - tile_side/2)
		y = int(starty + y - tile_side/2)
		img = read_class.getTile(level, x, y, tile_side, tile_side).array
		img = plot_one_box([int(tile_side/2-w/2), int(tile_side/2-h/2), int(tile_side/2+w/2), int(tile_side/2+h/2)], img, label='%.5f'%c, color=color)
		cv2.imwrite(os.path.join(save_root, '%s (%s)' % (slide, label), 'anchor%d-%d,%d-%.6f.jpg'%(j, x, y, c)), img)


def main(mtag, name, fig, ax, mlabel=None, marker=None, show_img=True):
	top_n = 100
	bottom_n = 0
	one_term_len = 50
	cls_type = 'transformer'
	list_path = r'D:\WSI_analysis\rnn\data_sets\visual_list.txt'
	if 'icn' in mtag:
	    anchor_num, model_dim = 3, 7
	if 'yolov3' in mtag:
	    anchor_num, model_dim = 2, 7
	if 'tiny' in mtag:
	    anchor_num, model_dim = 3, 6
	if 'yolov4' in mtag:
	    anchor_num, model_dim = 2, 7
	# =====================================================================================================================
	model_name = '%s_%s-mixed-top%d' % (cls_type, mtag, top_n)
	input_dim, n_layers, hidden_dim = get_params(name, model_name)
	thres = {'icn': {'transformer': 0.69, 'lstm': 0.85}, 'yolotiny': {'transformer': 0.49, 'lstm': 0.44}, 'yolov3': {'transformer': 0.71, 'lstm': 0.22}}
	model_outc = {'transformer': 0, 'lstm': -1, 'rnn': -1}
	data_root = 'sfyall'
	# =====================================================================================================================
	Data = SequenceDataset(r'F:\sfy1&2_2wfm', list_path, r'D:\sfy1&2_yolos', name=name, top_n=top_n, bottom_n=bottom_n, balance=False, dis_thers=0, joint=0, data_root=data_root)

	print(len(Data), 'Data loaded')

	center_side = 41600
	tile_side = 1024
	save_root = os.path.join(r'D:\WSI_analysis\mechine_learning\vis', '%s-side%d-topn%d-withBbox-max100'%(mtag,tile_side,top_n))
	color = [255, 0, 0]
	out = {}
	for i, data in enumerate(Data):
		if data['outputs'] is None: continue
		starty, startx = whlist[data['info']]
		level = levellist[data['info']]
		startx = (startx - center_side) / 2
		starty = (starty - center_side) / 2
		label = data['label']
		outputs = data['outputs']
		if 'tiny' not in mtag:
			confs = torch.sigmoid(outputs[..., 4] * torch.exp(-1 * outputs[..., 5]))  # conf
			fp_confs = torch.sigmoid(outputs[..., 5])  # fp_conf
		elif 'mnv2' not in mtag:
			confs = torch.sigmoid(outputs[..., 4])
			fp_confs = [None for _ in range(confs.shape[0])]
		else:
			confs = torch.sigmoid(outputs)
			fp_confs = [None for _ in range(confs.shape[0])]
		os.makedirs(os.path.join(save_root, '%s (%s)' % (slide2ind[data['info']], label)), exist_ok=True)
		path = slide2path[data['info']]
		read_class = fast_read.ReadClass(path, False)
		bx_maxs, by_maxs, conf_maxs = [], [], []
		for j in range(confs.shape[0]):
			conf, fp_conf = confs[j], fp_confs[j]
			_max, _min = sort_output(conf, one_term_len=one_term_len)
			# if 'tiny' not in mtag:
			# 	fp_max, fp_min = sort_output(fp_conf, one_term_len=int(top_n/2))
			if 'mnv2' not in mtag:
				anchors = Anchors[mtag][j]
				grid_x, grid_y, anchor_w, anchor_h, stride_w, stride_h = compute_grid_offsets(outputs.shape[-3:-1], [o * 32 for o in outputs.shape[-3:-1]], anchors)
				ox = torch.sigmoid(outputs[j,:,:,:,0])  # Center x
				oy = torch.sigmoid(outputs[j,:,:,:,1])  # Center y
				bw = outputs[j,:,:,:,2]  # Width
				bh = outputs[j,:,:,:,3]  # Height
				bx = (ox.data + grid_x) * stride_w
				by = (oy.data + grid_y) * stride_h
				bw = (torch.exp(bw.data) * anchor_w) * stride_w
				bh = (torch.exp(bh.data) * anchor_h) * stride_h
				bx_max = bx[0].reshape(-1)[_min]
				by_max = by[0].reshape(-1)[_min]
				bw_max = bw[0].reshape(-1)[_min]
				bh_max = bh[0].reshape(-1)[_min]
			else:
				grid_x, grid_y, anchor_w, anchor_h, stride_w, stride_h = compute_grid_offsets(outputs.shape[-2:], [o * 320 for o in outputs.shape[-2:]], [[1,1]])
				bx = grid_x * stride_w
				by = grid_y * stride_h
				bx_max = bx[0].reshape(-1)[_min]
				by_max = by[0].reshape(-1)[_min]
			conf_max = conf.reshape(-1)[_min]
			bx_maxs += [bx_max[:int(one_term_len/confs.shape[0])]]
			by_maxs += [by_max[:int(one_term_len/confs.shape[0])]]
			conf_maxs += [conf_max[:int(one_term_len/confs.shape[0])]]
			# save_bbox(startx, starty, bx_max, by_max, bw_max, bh_max, conf_max, read_class, save_root, slide2ind[data['info']], label, tile_side, color, level, j)
		contraction_wsi(startx, starty, whlist[data['info']], torch.cat(bx_maxs), torch.cat(by_maxs), torch.cat(conf_maxs), read_class, mlabel, marker, "%d: %s"%(slide2ind[data['info']], 'pos' if label == 1 else 'neg'), fig, ax[slide2ind[data['info']]], show_img, level)

	ax[-1].scatter([],[],c='k',alpha=0.3,marker=marker,linewidths=0,s=50,label=mlabel)


if __name__ == '__main__':
	os.environ['CUDA_VISIBLE_DEVICES'] = '2'

	fig, ax = plt.subplots(1, 10, gridspec_kw = {'width_ratios':[4 for _ in range(10)]}, figsize=(40,1))
	ax = ax.flatten()
	fig, ax = None, None
	mtag, name = 'yolotiny', 'yolotinydisconserve0_forvis'
	main(mtag, name, fig, ax, mlabel='Tiny', marker='X')
	mtag, name = 'mnv2', 'mnv2disconserve0_forvis'
	main(mtag, name, fig, ax, mlabel='MNV2', marker='^')
	mtag, name = 'yolov3', 'yolov3NewXYconverseoverlap288_forvis'
	main(mtag, name, fig, ax, mlabel='YOLOv3', marker='P', show_img=False)
	mtag, name = 'icn', 'icndisconserve0_forvis' 
	main(mtag, name, fig, ax, mlabel='YOLCO', marker='o', show_img=False)
	cmap = plt.cm.get_cmap('rainbow', 100)# this is the colormap used to display the spectrogram
	norm = mpl.colors.Normalize(vmin=0, vmax=1) # these are the min and max values from the spectrogram
	fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='probabilities', ax=ax[-1])
	fig.legend(scatterpoints=1,frameon=False,labelspacing=1)
	plt.show()
	exit()