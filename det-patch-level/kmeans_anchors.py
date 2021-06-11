from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt

label_root = 'Z:/wei/All_Of_Them/labels1024/'

for r,d, labels_n in os.walk(label_root):
    break
Xs = []
for label in tqdm(labels_n):
    
    with open(label_root + label) as f:
        ls = f.read().split('\n')[:-1]
    ls = [l.split(' ') for l in ls][:-1]
    X = []

    for score, cx, cy, w, h in ls:
        X.append([float(w), float(h)])
    Xs.extend(X)

print(len(Xs))
Xs = np.array(Xs)
kmeans = KMeans(n_clusters=6).fit(Xs)
X_label = kmeans.labels_
X_center = kmeans.cluster_centers_
print(X_center*1024)

anchor1 = np.where(X_label==0)[0]
# anchor1 = anchor1[:int(len(anchor1)/10)]
plt.scatter(Xs[anchor1, 0], Xs[anchor1, 1], color='red', s=10)
plt.scatter(X_center[0, 0], X_center[0, 1], color='yellow', s=40, marker='x')

anchor2 = np.where(X_label==1)[0]
# anchor2 = anchor2[:int(len(anchor2)/10)]
plt.scatter(Xs[anchor2, 0], Xs[anchor2, 1], color='green', s=10)
plt.scatter(X_center[1, 0], X_center[1, 1], color='yellow', s=40, marker='x')

anchor3 = np.where(X_label==2)[0]
# anchor3 = anchor3[:int(len(anchor3)/10)]
plt.scatter(Xs[anchor3, 0], Xs[anchor3, 1], color='darkgreen', s=10)
plt.scatter(X_center[2, 0], X_center[2, 1], color='yellow', s=40, marker='x')

anchor4 = np.where(X_label==3)[0]
# anchor4 = anchor4[:int(len(anchor4)/10)]
plt.scatter(Xs[anchor4, 0], Xs[anchor4, 1], color='blue', s=10)
plt.scatter(X_center[3, 0], X_center[3, 1], color='yellow', s=40, marker='x')

anchor5 = np.where(X_label==4)[0]
# anchor5 = anchor5[:int(len(anchor5)/10)]
plt.scatter(Xs[anchor5, 0], Xs[anchor5, 1], color='black', s=10)
plt.scatter(X_center[4, 0], X_center[4, 1], color='yellow', s=40, marker='x')

anchor6 = np.where(X_label==5)[0]
# anchor6 = anchor6[:int(len(anchor6)/10)]
plt.scatter(Xs[anchor6, 0], Xs[anchor6, 1], color='lime', s=10)
plt.scatter(X_center[5, 0], X_center[5, 1], color='yellow', s=40, marker='x')


plt.show()
