import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from generate_dataset import *

training_size = 100
testing_size = 100


x_start = -10
x_end = 10
y_start = -10
y_end = 10
margin = 0

KERNEL = 'linear'

training_data, training_models = make_poly(training_size, x_start, x_end, y_start, y_end, margin)

testing_data, testing_models = make_poly(testing_size, x_start, x_end, y_start, y_end, margin)


x_min = np.array(testing_data)[0:, 0].min()
x_max = np.array(testing_data)[:, 0].max()
y_min = np.array(testing_data)[:, 1].min()
y_max = np.array(testing_data)[:, 1].max()



clf = svm.SVC(kernel=KERNEL, C=1.0)


#clf = svm.LinearSVC()
print("kernel: ", KERNEL)
print("fitting...")
new_models = clf.fit(training_data, training_models).predict(testing_data)
print("done fitting")
error_count = 0



fig1, ax1 = plt.subplots()
ax1.set_title("Classified")


fig2, ax2 = plt.subplots()
ax2.set_title("Non-Linear Dataset")
for i in range(testing_size):
    #print feedback every 100 input
    if i % 10 == 0:
        print("Plotting: ", i)
    #Plotting original inputs
    if(testing_models[i] == 1):
        ax2.plot(testing_data[i][0], testing_data[i][1], 'ro')
    else:
        ax2.plot(testing_data[i][0], testing_data[i][1], 'bo')
    #Plotting classified input
    if(new_models[i] == 1):
        ax1.plot(testing_data[i][0], testing_data[i][1], 'ro')
    else:
        ax1.plot(testing_data[i][0], testing_data[i][1], 'bo')
    #calculating error
    if(testing_models[i] != new_models[i]):
        error_count += 1

print(error_count/len(testing_data) * 100)
#original input divided line
#ax2.plot([x_min, x_max], [x_min, x_max])
#classified input divide line
XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
Z = Z.reshape(XX.shape)
ax1.contour(XX, YY, Z, colors=['k', 'k', 'k'],
            linestyles=['--', '-', '--'], levels=[-(abs(margin-0.1)/10), 0, (abs(margin+0.1)/10)])
# Put the result into a color plot
plt.show()
