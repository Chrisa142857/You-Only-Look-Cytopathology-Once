from sklearn import svm
from sklearn.model_selection import GridSearchCV
import joblib 
import os

from svm_dataset import SequenceDataset

def svc_train(x, y, xtest, ytest, mtag, top_n):
	m_name = '%s-top%d-svc_best_nosearch_linear' % (mtag, top_n)
	# if not os.path.exists('%s.joblib'%m_name):
	clf = svm.LinearSVC(random_state=142857)
	clf.fit(x, y)
	joblib.dump(clf, '%s.joblib'%m_name) 
	# else:
	# 	clf = joblib.load('%s.joblib'%m_name)
	csv_n = 'v4_svm_%s-mixed_r5-top100_transformer_pred_val.csv' % mtag
	with open(csv_n, 'w') as f:
	    f.write('probs,labels\n')

	preds = clf.predict(xtest)
	wrong = []
	for p, l in zip(preds, ytest):
		with open(csv_n, 'a') as f:
		    f.write('%f,%f\n'% (p, l))
		if p != l: 
			wrong += [p]
	tpr = 1-(len([p for p in wrong if p == 1])/len(ytest))
	tnr = 1-(len([p for p in wrong if p == 0])/len(ytest))
	acc = (tpr+tnr) / 2
	return acc, tpr, tnr

def get_data(t_list_path, v_list_path, test_list_path, name, top_n, bottom_n):
	x, y = [], []

	Data = SequenceDataset(r'F:\sfy1&2_2wfm', t_list_path, r'D:\sfy1&2_yolos', name=name, top_n=top_n, bottom_n=bottom_n, balance=False, dis_thers=0, joint=0, data_root='sfyall')
	for i, data in enumerate(Data):
		x += [data['data'].reshape(-1).numpy()]
		y += [data['label']]

	Data = SequenceDataset(r'F:\sfy1&2_2wfm', v_list_path, r'D:\sfy1&2_yolos', name=name, top_n=top_n, bottom_n=bottom_n, balance=False, dis_thers=0, joint=0, data_root='sfyall')
	for i, data in enumerate(Data):
		x += [data['data'].reshape(-1).numpy()]
		y += [data['label']]

	xtest, ytest = [], []
	Data = SequenceDataset(r'F:\sfy1&2_2wfm', test_list_path, r'D:\sfy1&2_yolos', name=name, top_n=top_n, bottom_n=bottom_n, balance=False, dis_thers=0, joint=0, data_root='sfyall')
	for i, data in enumerate(Data):
		xtest += [data['data'].reshape(-1).numpy()]
		ytest += [data['label']]
	return x, y, xtest, ytest

# =====================================================================================================================
# t_list_path = r'D:\WSI_analysis\rnn\data_sets\sfy12_train.txt'
# v_list_path = r'D:\WSI_analysis\rnn\data_sets\sfy12_val.txt'
# test_list_path = r'D:\WSI_analysis\rnn\data_sets\sfy12_test.txt'

t_list_path = r'D:\WSI_analysis\rnn\data_sets\sfyall_train.txt'
v_list_path = r'D:\WSI_analysis\rnn\data_sets\sfyall_val.txt'
test_list_path = r'D:\WSI_analysis\rnn\data_sets\sfyall_test.txt'
# =====================================================================================================================
accs, tprs, tnrs, names = [], [], [], []
bottom_n = 0
# name = 'icndisconserve0' 
# name = 'yolov3NewXYconverseoverlap288'
# name = 'yolotinydisconserve0'
# =====================================================================================================================
# top_n = 100
# name = 'icndisconserve2' 
# x, y, xtest, ytest = get_data(t_list_path, v_list_path, test_list_path, name, top_n, bottom_n)
# acc, tpr, tnr = svc_train(x, y, xtest, ytest, name, top_n)
# accs += [acc]
# tprs += [tpr]
# tnrs += [tnr]
# names += [name+'-top%d'%top_n]
# # =====================================================================================================================
# top_n = 100
# name = 'icndisconserve20' 
# x, y, xtest, ytest = get_data(t_list_path, v_list_path, test_list_path, name, top_n, bottom_n)
# acc, tpr, tnr = svc_train(x, y, xtest, ytest, name, top_n)
# accs += [acc]
# tprs += [tpr]
# tnrs += [tnr]
# names += [name+'-top%d'%top_n]
# # =====================================================================================================================
# top_n = 100
# name = 'icndisconserve100' 
# x, y, xtest, ytest = get_data(t_list_path, v_list_path, test_list_path, name, top_n, bottom_n)
# acc, tpr, tnr = svc_train(x, y, xtest, ytest, name, top_n)
# accs += [acc]
# tprs += [tpr]
# tnrs += [tnr]
# names += [name+'-top%d'%top_n]
# # =====================================================================================================================
# top_n = 1000
# name = 'icndisconserve_withWH' 
# x, y, xtest, ytest = get_data(t_list_path, v_list_path, test_list_path, name, top_n, bottom_n)
# acc, tpr, tnr = svc_train(x, y, xtest, ytest, name, top_n)
# accs += [acc]
# tprs += [tpr]
# tnrs += [tnr]
# names += [name+'-top%d'%top_n]
# =====================================================================================================================
# top_n = 10
# name = 'icndisconserve0' 
# x, y, xtest, ytest = get_data(t_list_path, v_list_path, test_list_path, name, top_n, bottom_n)
# acc, tpr, tnr = svc_train(x, y, xtest, ytest, 'icnsfy12', top_n)
# accs += [acc]
# tprs += [tpr]
# tnrs += [tnr]
# names += [name+'-top%d'%top_n]
# # =====================================================================================================================
# top_n = 20
# name = 'icndisconserve0' 
# x, y, xtest, ytest = get_data(t_list_path, v_list_path, test_list_path, name, top_n, bottom_n)
# acc, tpr, tnr = svc_train(x, y, xtest, ytest, 'icnsfy12', top_n)
# accs += [acc]
# tprs += [tpr]
# tnrs += [tnr]
# names += [name+'-top%d'%top_n]
# =====================================================================================================================
# top_n = 50
# name = 'icndisconserve0' 
# x, y, xtest, ytest = get_data(t_list_path, v_list_path, test_list_path, name, top_n, bottom_n)
# acc, tpr, tnr = svc_train(x, y, xtest, ytest, 'icnsfy12', top_n)
# accs += [acc]
# tprs += [tpr]
# tnrs += [tnr]
# names += [name+'-top%d'%top_n]
# =====================================================================================================================
top_n = 100
name = 'mnv2disconserve0' 
x, y, xtest, ytest = get_data(t_list_path, v_list_path, test_list_path, name, top_n, bottom_n)
acc, tpr, tnr = svc_train(x, y, xtest, ytest, name, top_n)
accs += [acc]
tprs += [tpr]
tnrs += [tnr]
names += [name+'-top%d'%top_n]
# =====================================================================================================================
top_n = 100
name = 'icndisconserve0' 
x, y, xtest, ytest = get_data(t_list_path, v_list_path, test_list_path, name, top_n, bottom_n)
acc, tpr, tnr = svc_train(x, y, xtest, ytest, name, top_n)
accs += [acc]
tprs += [tpr]
tnrs += [tnr]
names += [name+'-top%d'%top_n]
# =====================================================================================================================
top_n = 100
name = 'yolotinydisconserve0' 
x, y, xtest, ytest = get_data(t_list_path, v_list_path, test_list_path, name, top_n, bottom_n)
acc, tpr, tnr = svc_train(x, y, xtest, ytest, name, top_n)
accs += [acc]
tprs += [tpr]
tnrs += [tnr]
names += [name+'-top%d'%top_n]
# =====================================================================================================================
top_n = 100
name = 'yolov3NewXYconverseoverlap288' 
x, y, xtest, ytest = get_data(t_list_path, v_list_path, test_list_path, name, top_n, bottom_n)
acc, tpr, tnr = svc_train(x, y, xtest, ytest, name, top_n)
accs += [acc]
tprs += [tpr]
tnrs += [tnr]
names += [name+'-top%d'%top_n]
# =====================================================================================================================
# top_n = 150
# name = 'icndisconserve0' 
# x, y, xtest, ytest = get_data(t_list_path, v_list_path, test_list_path, name, top_n, bottom_n)
# acc, tpr, tnr = svc_train(x, y, xtest, ytest, 'icnsfy12', top_n)
# accs += [acc]
# tprs += [tpr]
# tnrs += [tnr]
# names += [name+'-top%d'%top_n]
# =====================================================================================================================
# top_n = 200
# name = 'icndisconserve0' 
# x, y, xtest, ytest = get_data(t_list_path, v_list_path, test_list_path, name, top_n, bottom_n)
# acc, tpr, tnr = svc_train(x, y, xtest, ytest, 'icnsfy12', top_n)
# accs += [acc]
# tprs += [tpr]
# tnrs += [tnr]
# names += [name+'-top%d'%top_n]
# =====================================================================================================================
print('    ', names)
print('acc,', accs)
print('tpr,', tprs)
print('tnr,', tnrs)
