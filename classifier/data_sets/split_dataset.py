import MySQLdb
import os

def prepareLabels(label_lines):
    pos_label = ['Positive', 'pos', '_p', 'sfy9']
    neg_label = ['Negative', 'neg']
    slideLable = {}
    posn = 0
    negn = 0
    posSlides = []
    negSlides = []
    for line in label_lines:
        file = line.split('\\')[-1].split('/')[-1]
        slide = ''
        for s in file.split('.')[:-1]:
            slide += s + '.'
        slide = slide[:-1]
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

## count of each group
# with open('sfy_all_slide_list_train.txt', 'r') as f:
#  t=f.read().split('\n')[:-1]
# with open('sfy_all_slide_list_val.txt', 'r') as f:
#  v=f.read().split('\n')[:-1]
# ls = t+v
# # tag = '3DHisTech'
# # tag = 'SZSQ'
# tag = 'WNLO'
# xs = [l for l in ls if tag in l and ('sfy6' in l or 'sfy7' in l or 'sfy8' in l)]

# _, pn, nn, _, _ = prepareLabels(xs)
# print(len(xs), pn, nn)
# exit()
###########
all_slides = []
for r,d,f in os.walk(r'O:\WSIData\TransSRP'):
	all_slides += [os.path.join(r,fi) for fi in f if fi.endswith('.srp')]
db = MySQLdb.connect("192.168.0.160", 'weiziquan','123','das',charset='utf8')
cursor = db.cursor()

with open('sfy_all_slide_list_train.txt', 'r') as f:
	ts = f.read().split('\n')[:-1]
with open('sfy_all_slide_list_val.txt', 'r') as f:
	vs = f.read().split('\n')[:-1]
with open('sfy_manualcheck_slide_list.txt', 'r') as f:
	ms = f.read().split('\n')[:-1]

gs = ['Shengfuyou_8th','Shengfuyou_7th','Shengfuyou_6th']
xywgs = []
tnum = 0
vnum=0
wrong = 0
sfynum=0
nonenum = 0
sps = []
all_gs = []
transsrp = {}
for m in ms:
	m=m.replace('.sdpc', '')
	transsrp[m] = [p for p in all_slides if m in p]
	print('')
	print(m.replace('.sdpc', ''))
	cursor.execute('SELECT slide_group, slide_path FROM das.slides where slide_name like "{}%"'.format(m))
	results = cursor.fetchall()
	if len(results) == 0: 
		nonenum+= 1
	all_gs += [r[0] for r in results if r[0] not in all_gs]
	sp = [r[1] for r in results if r[0] in gs]
	if len(sp) > 0:
		if len([t for t in ts if m in t]) != 0:
			sps.append([t for t in ts if m in t]) 
		elif len([v for v in vs if m in v]) != 0:
			sps.append([v for v in vs if m in v]) 
		sfynum+=1
	print(results)
	f1, f2 = False, False
	if len([t for t in ts if m in t]) == 0:
		tnum += 1
		f1 = True
		print('not train', m)
	if len([v for v in vs if m in v]) == 0:
		vnum += 1
		f2 = True
		print('not val', m)
	if f1 and f2:
		wrong += 1
		print('wrong', m)

have_seq = ''
no_seq = ''
manual_list = ''
for k in transsrp:
	s = transsrp[k]
	manual_list += s[0]+'\n'
	if len([si for si in s if si in ts or si in vs]) > 0:
		have_seq += s[0]+'\n'
	else:
		no_seq += s[0]+'\n'
# with open('sfy_have_seq_manualcheck_list.txt', 'w') as f:
# 	f.write(have_seq)

# with open('sfy_no_seq_manualcheck_list.txt', 'w') as f:
# 	f.write(no_seq)

with open('manualcheck_srp_list.txt', 'w') as f:
	f.write(manual_list)
print(transsrp)
print('')
print(sps, len(sps))
print('')
print(all_gs)
print(tnum, vnum, wrong, sfynum, nonenum)
print('len(all_slides)', len(all_slides))
