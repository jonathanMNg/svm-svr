import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.datasets import make_blobs

N = 100

X_axis = np.random.randint(-100,100, N)
Y_axis = np.random.randint(-100,100, N)
data_pts = np.array(list(zip(X_axis,Y_axis)))

X, y = make_blobs(n_samples=1000, centers=2, random_state=6)
clf = svm.SVC(kernel='poly', C=1, degree=2)
clf.fit(X,y)

models = clf.predict(data_pts)

clf.fit(data_pts,models)
models = clf.predict(data_pts)

plt.scatter(data_pts[:, 0], data_pts[:, 1], c=models, s=30, cmap=plt.cm.Paired)
ax=plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)

xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, colors='k', levels=[-3, 0, 3], alpha=0.5,
           linestyles=['--', '-', '--'])

ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none')

plt.show()
