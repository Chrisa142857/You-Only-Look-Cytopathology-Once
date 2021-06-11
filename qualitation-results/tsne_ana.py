from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import os
from dataset import str2slide, prepareLabels, SequenceDataset
from det_model import Darknet
from cls_model import create_model
import datetime
import joblib 

class trans_PARAM:
	def __init__(self):
		x = {
			'icn' : [['mixed', 'mixed_r1', 'mixed_r2', 'mixed_r3', 'mixed_r4', 'mixed_r5'], [294, 244, 304, 426, 261, 184]],
			'mnv2' : [['mixed', 'mixed_r1', 'mixed_r2', 'mixed_r3', 'mixed_r4', 'mixed_r5'],[579, 601, 387, 353, 527, 360]],
			'yolotiny' : [['mixed', 'mixed_r1', 'mixed_r2', 'mixed_r3', 'mixed_r4', 'mixed_r5'],[443, 647, 334, 492, 360, 523]],
			'yolov3' : [['mixed', 'mixed_r1', 'mixed_r2', 'mixed_r3', 'mixed_r4', 'mixed_r5'],[358, 138, 471, 600, 505, 362]]
		}
		self._max = {
			'icn' : 'mixed_r5',
			'mnv2' : 'mixed_r5',
			'yolotiny' : 'mixed_r5',
			'yolov3' : 'mixed_r5'
		}
		self.thres = {}
		for k in self._max:
			for r, t in zip(x[k][0], x[k][1]): 
				if r == self._max[k]: 
					self.thres[k] = t/1000
					break

class lstm_PARAM:
	def __init__(self):
		x = {
			'icn' : [['mixed','mixed_r1', 'mixed_r2', 'mixed_r3', 'mixed_r4', 'mixed_r5'], [500,507, 505, 554, 602, 422]],
			'mnv2' : [['mixed','mixed_r1', 'mixed_r2', 'mixed_r3', 'mixed_r4', 'mixed_r5'],[500,505, 505, 505, 505, 505]],
			'yolotiny' : [['mixed','mixed_r1', 'mixed_r2', 'mixed_r3', 'mixed_r4', 'mixed_r5'],[500,430, 415, 453, 405, 410]],
			'yolov3' : [['mixed','mixed_r1', 'mixed_r2', 'mixed_r3', 'mixed_r4', 'mixed_r5'],[500,722, 505, 657, 505, 505]]
		}
		self._max = {
			'icn' : 'mixed',
			'mnv2' : 'mixed',
			'yolotiny' : 'mixed',
			'yolov3' : 'mixed'
		}
		self.thres = {}
		for k in self._max:
			for r, t in zip(x[k][0], x[k][1]): 
				if r == self._max[k]: 
					self.thres[k] = t/1000
					break

def plot_tsne(x, y, inds, color, label, marker):
	for i, ind in enumerate(inds):
		plt.scatter(y[ind,0], y[ind,1], s=30, alpha=0.5, color=color[i], label=label[i], marker=marker[i], linewidths=0)

def choose_slide(y, fig):
	y2slide = {}
	for i, yi in enumerate(y):
		tag = "%.6f,%.6f" % (yi[0], yi[1])
		if tag not in y2slide: y2slide[tag] = []
		y2slide[tag].append(slides[i])

	def onclick(event):
		xdis, ydis = y[:, 0] - event.xdata, y[:, 1] - event.ydata
		dis = xdis**2 + ydis**2
		ind = dis.argmin()
		tag = "%.6f,%.6f" % (y[ind,0], y[ind,1])	
		for i, index in enumerate(indexes):
			if ind in index:
				group = labels[i]
				break
		print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f, slide=%s, group=%s,%s, index=%s' %
		      ('double' if event.dblclick else 'single', event.button,
		       event.x, event.y, event.xdata, event.ydata, y2slide[tag], group, [slide2group[s] for s in y2slide[tag]], ind))
	cid = fig.canvas.mpl_connect('button_press_event', onclick)
	plt.show()
	fig.canvas.mpl_disconnect(cid)
	exit()


def get_classifier(weights_path, in_channel):
	weights_path = "D:/WSI_analysis/det/sRMBv02/yolov3_custom_best.pth"
	Ks = [[] for k in m if m[k].shape[0] == in_channel and len(m[k].shape) == 4]
	ki = 0
	for k in m:
		if m[k].shape[0] == in_channel:
			if len(m[k].shape) == 4: ki += 1
			Ks[ki].append(k)
	cs = []
	for ks in Ks:
		c = torch.nn.Conv2d(m[ks[0]].shape[1], m[ks[0]].shape[0], 1)
		with torch.no_grad():
			for cp, k in zip(c.parameters(), ks):
				cp.copy_(m[k])
		cs.append(c)
	return cs


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_params(name, model):
	if 'transformer' in model:
		n_layers = 10
	else:
		n_layers = 1
	if 'mnv2' in name:
	    input_dim = 1280
	elif 'icn' in name:
	    input_dim = 768
	elif 'yolotiny' in name:
	    input_dim = 768
	elif 'yolov3' in name:
	    input_dim = 1792

	return input_dim, n_layers, 2048

def main(cls_type, mtag, name, top_n, bottom_n, group, isSVM, ax=None, mlabel=None, color=None, isCurve=None):
	selected_ind = [66, 1462, 1354, 1834, 713, 879, 448, 866, 810, 585]
	ind2id = {}
	for i, ind in enumerate(selected_ind):
		ind2id[ind] = i
	# =====================================================================================================================
	list_path = r'D:\WSI_analysis\rnn\data_sets\%s_all_list.txt' % group
	# list_path = r'D:\WSI_analysis\rnn\data_sets\sfy_all_slide_list_val.txt'
	# =====================================================================================================================
	data_root = 'sfyall'
	with open(list_path, 'r') as f:
		ls = f.read().split('\n')[:-1]
	slide2group = {}
	for l in ls:
		if 'sfy1' in l and '3DHisTech' in l: 
			slide2group[str2slide(l.split('\\')[-1].split('/')[-1])] = 0
		elif ('sfy2' in l or 'sfy3' in l) and ('3DHisTech' in l): 
			slide2group[str2slide(l.split('\\')[-1].split('/')[-1])] = 2
		elif 'SZSQ' in l:
			slide2group[str2slide(l.split('\\')[-1].split('/')[-1])] = 4
		elif 'WNLO' in l:
			slide2group[str2slide(l.split('\\')[-1].split('/')[-1])] = 6
	# =====================================================================================================================
	if not isSVM:
		model_outc = {'transformer': 0, 'lstm': -1, 'rnn': -1}
		thres = {'transformer': trans_PARAM().thres[mtag], 'lstm': lstm_PARAM().thres[mtag]}
		run = {'transformer': trans_PARAM()._max[mtag], 'lstm': lstm_PARAM()._max[mtag]}
		input_dim, n_layers, hidden_dim = get_params(name, cls_type)
		model_name = '%s_%s-%s-top%d' % (cls_type, mtag, run[cls_type], top_n)
		model = create_model(model_name)(input_dim=input_dim, batch_size=1, n_layers=n_layers, hidden_dim=hidden_dim).cuda()
		try:
			resume = torch.load('D:\\WSI_analysis\\rnn_output\\v4_%s_checkpoint_best.pth' % model_name)
		except:
			resume = torch.load('W:\\WSI_analysis\\rnn_output\\v4_%s_checkpoint_best.pth' % model_name)
		model.load_state_dict(resume['state_dict'])
		hidden = model.init_hidden()
		model.eval()
	else:
		m_name = '%s-top%d-svc_best_nosearch_linear' % (mtag, top_n)
		clf = joblib.load('%s.joblib'%m_name)
	# =====================================================================================================================
	if not os.path.exists("tsne-%s.pth"% model_name if not isSVM else m_name) or True:
		print('%s Data' % group)
		Data = SequenceDataset(r'F:\sfy1&2_2wfm', list_path, r'D:\sfy1&2_yolos', name=name, top_n=top_n, bottom_n=bottom_n, balance=False, dis_thers=0, joint=0, data_root=data_root)
		leng = 4
		lenseq = int(top_n/10)
		indexes = [[] for _ in range(2*leng)]
		seqs = []
		markers = ['o', 'o', 's', 's', '^', '^', 'v', 'v']
		labels = ['s1-n', 's1-p', 's2-n', 's2-p', 's3-n', 's3-p', 's4-n', 's4-p']
		cmap = plt.cm.get_cmap('tab20', 20)
		# colors = [cmap(i) for i in range(2*leng)]
		colors = []
		for i in range(leng):
			colors += ['lightgrey', cmap(2*i)]
		wrong_ind = [[], []]
		wrong_label = ['wrong-p', 'wrong-n']
		wrong_color = ['black', 'lightgrey']
		wrong_marker = ['.', '.']
		start = datetime.datetime.now()
		fpn = 0
		fnn = 0
		slides = []
		for i, data in enumerate(Data):
			if i not in selected_ind and isCurve: continue
			if not isSVM:
				with torch.no_grad():
					if 'transformer' not in model_name:
						o, hidden, v = model(torch.stack([data['data']]).cuda(), hidden)
						hidden = repackage_hidden(hidden)
					else:
						o, v = model(torch.stack([data['data']]).cuda())
				o = o[0].cpu()
				v = v.cpu()
# =========================================================================================================
#				折綫圖
				if isCurve:
					ax[ind2id[i]].plot([iii for iii in range(100)], o[-100:].numpy()[::-1], label=mlabel, color=color[0])
					ax[ind2id[i]].plot([0, 99], [thres[cls_type], thres[cls_type]], '--', color=color[1], label=None)
					ax[ind2id[i]].set_ylim(-0.01, 1.01)
					ax[ind2id[i]].set_xlim(0, 100)
					ax[ind2id[i]].set_xlabel('top-x')
					ax[ind2id[i]].set_ylabel('probabilities')
					ax[ind2id[i]].set_title('%d:%s' % (ind2id[i], 'pos' if data['label'] == 1 else 'neg'))
# =========================================================================================================

				o[o>=thres[cls_type]] = 1
				o[o<thres[cls_type]] = 0
				o = o.int()
				seqs += [v[:, model_outc[cls_type]]]
				out = o[model_outc[cls_type]]
				label = data['label']
				if isCurve: print(mtag, data['info'], data['label'], out, i, ind2id[i])
			else:
				seqs += [data['data'].reshape(1, -1)]
				label = data['label']
				# indexes[slide2group[data['info']]+label] += [i]
				out = clf.predict(data['data'].reshape(1, -1).numpy())[0]

			if out == label:
				g = slide2group[data['info']]
				# indexes[g+out] += [j for j in range(i*lenseq, (i+1)*lenseq)]
				# for j, ji in zip(range(i*lenseq, (i+1)*lenseq), range(lenseq)):
				# 	indexes[g+o[ji]] += [j]
				indexes[g+out] += [i]
			else:
				# wrong_ind += [j for j in range(i*lenseq, (i+1)*lenseq)]
				wrong_ind[out] += [i]
				if label == 1: fnn += 1
				if label == 0: fpn += 1
			slides.append(data['info'])
		print(datetime.datetime.now()-start)
		indexes += wrong_ind
		colors += wrong_color
		labels += wrong_label
		markers += wrong_marker
		acc = 1-((len(indexes[-1])+len(indexes[-2]))/len(Data))
		fpr = fpn/len(Data)
		fnr = fnn/len(Data)
		seqs = torch.cat(seqs)
		tsne = TSNE(random_state=142857)
		y = tsne.fit_transform(seqs)
		if not isCurve: torch.save({'tsne': torch.from_numpy(y), 'seqs': seqs, 'indexes': indexes, "colors": colors, 'labels': labels, 'markers': markers, 'acc': acc, 'fpr': fpr, 'fnr': fnr}, "tsne-%s.pth"% model_name if not isSVM else m_name)
	else:
		x = torch.load("tsne-%s.pth"% model_name if not isSVM else m_name)
		y = x['tsne']
		seqs = x['seqs']
		indexes = x['indexes']
		colors = x['colors']
		labels = x['labels']
		markers = x['markers']

	return seqs, y, indexes, colors, labels, markers

def plot_fig(seqs, y, indexes, colors, labels, markers):
	plot_tsne(seqs, y, indexes, colors, labels, markers)
	# plt.legend()
	# choose_slide(y, fig)
	selected_ind = [66, 1462, 1354, 1834, 713, 879, 448, 866, 810, 585]
	# plt.scatter(y[selected_ind, 0], y[selected_ind, 1], s=30, color='black', marker='*')
	for i, ind in enumerate(selected_ind):
		# plt.text(y[ind, 0], y[ind, 1], str(ind))
		plt.annotate("%d" % i, xy=(y[ind, 0], y[ind, 1]), xytext=(-3, -3), textcoords='offset points', color='white', bbox=dict(boxstyle='round,pad=0.1', fc='black', ec='k', lw=0, alpha=0.3)) #, arrowprops=dict(arrowstyle='-|>',connectionstyle='arc3',color='black',alpha=0.5)

def tsne_pt():
	top_n = 100
	bottom_n = 0
	group = 'all'
	plt.figure(figsize=(15, 15))
	plt.subplots_adjust(top=1, bottom=0, left=0.1, right=0.9)
	# =====================================================================================================================
	cls_type = 'lstm'
	isSVM = True
	mtag, name = 'mnv2', 'mnv2disconserve0' 
	seqs, y, indexes, colors, labels, markers = main(cls_type, mtag, name, top_n, bottom_n, group, isSVM)
	plt.subplot(221)
	plt.ylabel('MNV2')
	plot_fig(seqs, y, indexes, colors, labels, markers)
	plt.title('SVM')
	mtag, name = 'icn', 'icndisconserve0' 
	seqs, y, indexes, colors, labels, markers = main(cls_type, mtag, name, top_n, bottom_n, group, isSVM)
	plt.subplot(224)
	plt.ylabel('YOLCO')
	plot_fig(seqs, y, indexes, colors, labels, markers)
	mtag, name = 'yolov3', 'yolov3NewXYconverseoverlap288'
	seqs, y, indexes, colors, labels, markers = main(cls_type, mtag, name, top_n, bottom_n, group, isSVM)
	plt.subplot(223)
	plt.ylabel('YOLOv3')
	plot_fig(seqs, y, indexes, colors, labels, markers)
	mtag, name = 'yolotiny', 'yolotinydisconserve0'
	seqs, y, indexes, colors, labels, markers = main(cls_type, mtag, name, top_n, bottom_n, group, isSVM)
	plt.subplot(222)
	plt.ylabel('Tiny')
	plot_fig(seqs, y, indexes, colors, labels, markers)
	plt.show()
	# =====================================================================================================================
	# plt.figure(figsize=(15, 15))
	# plt.subplots_adjust(top=1, bottom=0, left=0.1, right=0.9)
	# # =====================================================================================================================
	# cls_type = 'lstm'
	# isSVM = False
	# mtag, name = 'mnv2', 'mnv2disconserve0' 
	# seqs, y, indexes, colors, labels, markers = main(cls_type, mtag, name, top_n, bottom_n, group, isSVM)
	# plt.subplot(221)
	# plot_fig(seqs, y, indexes, colors, labels, markers)
	# plt.title('LSTM')
	# plt.ylabel('MNV2')
	# mtag, name = 'icn', 'icndisconserve0' 
	# seqs, y, indexes, colors, labels, markers = main(cls_type, mtag, name, top_n, bottom_n, group, isSVM)
	# plt.subplot(224)
	# plt.ylabel('YOLCO')
	# plot_fig(seqs, y, indexes, colors, labels, markers)
	# mtag, name = 'yolov3', 'yolov3NewXYconverseoverlap288'
	# seqs, y, indexes, colors, labels, markers = main(cls_type, mtag, name, top_n, bottom_n, group, isSVM)
	# plt.subplot(223)
	# plt.ylabel('YOLOv3')
	# plot_fig(seqs, y, indexes, colors, labels, markers)
	# mtag, name = 'yolotiny', 'yolotinydisconserve0'
	# seqs, y, indexes, colors, labels, markers = main(cls_type, mtag, name, top_n, bottom_n, group, isSVM)
	# plt.subplot(222)
	# plt.ylabel('Tiny')
	# plot_fig(seqs, y, indexes, colors, labels, markers)
	# plt.show()
	# =====================================================================================================================
	# plt.figure(figsize=(15, 15))
	# plt.subplots_adjust(top=1, bottom=0, left=0.1, right=0.9)
	# # =====================================================================================================================
	# cls_type = 'transformer'
	# isSVM = False
	# mtag, name = 'mnv2', 'mnv2disconserve0' 
	# seqs, y, indexes, colors, labels, markers = main(cls_type, mtag, name, top_n, bottom_n, group, isSVM)
	# plt.subplot(221)
	# plot_fig(seqs, y, indexes, colors, labels, markers)
	# plt.title('Transformer')
	# plt.ylabel('MNV2')
	# mtag, name = 'icn', 'icndisconserve0' 
	# seqs, y, indexes, colors, labels, markers = main(cls_type, mtag, name, top_n, bottom_n, group, isSVM)
	# plt.subplot(224)
	# plt.ylabel('YOLCO')
	# plot_fig(seqs, y, indexes, colors, labels, markers)
	# mtag, name = 'yolov3', 'yolov3NewXYconverseoverlap288'
	# seqs, y, indexes, colors, labels, markers = main(cls_type, mtag, name, top_n, bottom_n, group, isSVM)
	# plt.subplot(223)
	# plt.ylabel('YOLOv3')
	# plot_fig(seqs, y, indexes, colors, labels, markers)
	# mtag, name = 'yolotiny', 'yolotinydisconserve0'
	# seqs, y, indexes, colors, labels, markers = main(cls_type, mtag, name, top_n, bottom_n, group, isSVM)
	# plt.subplot(222)
	# plt.ylabel('Tiny')
	# plot_fig(seqs, y, indexes, colors, labels, markers)
	# plt.show()
	# =====================================================================================================================
	# fig, ax = plt.subplots(10, 1, figsize=(40,40))
	# ax = ax.flatten()
	# cmap = plt.cm.get_cmap('tab20', 20)
	# # =====================================================================================================================
	# cls_type = 'transformer'
	# isSVM = False
	# isCurve = True
	# mtag, name, mlabel = 'icn', 'icndisconserve0', 'YOLCO'
	# seqs, y, indexes, colors, labels, markers = main(cls_type, mtag, name, top_n, bottom_n, group, isSVM, ax, mlabel, [cmap(0), cmap(1)], isCurve)
	# mtag, name, mlabel = 'yolov3', 'yolov3NewXYconverseoverlap288', 'YOLOv3'
	# seqs, y, indexes, colors, labels, markers = main(cls_type, mtag, name, top_n, bottom_n, group, isSVM, ax, mlabel, [cmap(2), cmap(3)], isCurve)
	# mtag, name, mlabel = 'yolotiny', 'yolotinydisconserve0', 'Tiny'
	# seqs, y, indexes, colors, labels, markers = main(cls_type, mtag, name, top_n, bottom_n, group, isSVM, ax, mlabel, [cmap(4), cmap(5)], isCurve)

	# # =====================================================================================================================
	# plt.legend()
	# plt.show()

	# fig.legend(scatterpoints=1,frameon=False,labelspacing=1)
	# plt.legend(scatterpoints=1,frameon=False,labelspacing=1)
	# fig.show()

if __name__ == '__main__':
	# =====================================================================================================================
	os.environ['CUDA_VISIBLE_DEVICES'] = '2'
	tsne_pt()
