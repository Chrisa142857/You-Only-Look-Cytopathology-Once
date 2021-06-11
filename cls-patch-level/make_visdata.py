import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
from slide_readtool import fast_read

slide = '16032022 yulan Ascus 4'

tag = 'train'
t=torch.load('D:/WSI_analysis/cls/data/det-%s-distinct.pth'%tag)

n=[]
for k in t:
 n+=[(x2-x1)*(y2-y1) for x1,y1,x2,y2 in t[k]]
num = 0
level = 8
for k in tqdm(t):
	reader = fast_read.ReadClass(k, False)
	h=reader.slide_height
	w=reader.slide_width
	img = reader.getTile(level, 0, 0, int(w/(2**level)), int(h/(2**level))).array
	side = 2**level
	re, th_img = cv2.threshold(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	fore = 255-torch.FloatTensor([th_img])
	num += len([f for f in fore.reshape(-1) if f == 255]) * side * side
	
print('bbox area', sum(n))
print('wsi area', num)
exit()		

side = 2080

for k in tqdm(t):
	if slide in k:
		reader = fast_read.ReadClass(k, False)
	else:
		continue
	for x1, y1, x2, y2 in t[k]:
		startx = np.random.randint(int(x2) - side, int(x1))
		starty = np.random.randint(int(y2) - side, int(y1))
		img = reader.getTile(0, startx, starty, side, side).array
		plt.figure(figsize=(13,13),dpi=160)
		plt.axis('off') 
		plt.gca().xaxis.set_major_locator(plt.NullLocator())
		plt.gca().yaxis.set_major_locator(plt.NullLocator())
		plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
		plt.margins(0,0)   
		plt.imshow(img[:, :, ::-1])

		plt.savefig('D:/WSI_analysis/cls/data/jpgs/%s-%d-%d-%d.jpg'%(slide, startx, starty, side))
		ax = plt.gca()
		for x1, y1, x2, y2 in t[k]:
			if startx<=x1<=startx+side and starty<=y1<=starty+side:
				ax.add_patch(plt.Rectangle((x1-startx, y1-starty), x2-x1, y2-y1, color="red", fill=False, linewidth=1))

		plt.savefig('D:/WSI_analysis/cls/data/jpgs_masked/%s-%d-%d-%d.jpg'%(slide, startx, starty, side))
		plt.close()


