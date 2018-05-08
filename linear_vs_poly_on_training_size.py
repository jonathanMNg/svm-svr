import numpy as np
import statistics
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from generate_dataset import *

n_samples = 100

samples_per_n = 100

testing_size = 500


x_start = -100
x_end = 100
y_start = -100
y_end = 100
margin = 2

svm_kernel = []
for kernel in (('linear', 'poly')):
    testing_data, testing_models = make_uniform(testing_size, x_start, x_end, y_start, y_end, margin)
    svm_data = []
    svm_acc_grow = 0
    for i in range(1, 11):
        training_size = i * n_samples
        svm_samples = []
        for __ in range(samples_per_n):
            training_data, training_models = make_uniform(training_size, x_start, x_end, y_start, y_end, margin)

            clf = svm.SVC(kernel=kernel, C=1.0)

            #fit data
            svm_results = clf.fit(training_data, training_models).predict(testing_data)
            #get the results model
            svm_acc = sum(1 if x == y else 0 for x, y in zip(svm_results, testing_models))
            svm_acc /= len(testing_data)
            svm_samples.append(svm_acc)
        svm_avg = statistics.mean(svm_samples) - svm_acc_grow
        svm_std = statistics.pstdev(svm_samples) - svm_acc_grow
        svm_data.append([training_size, svm_avg, svm_std, kernel])
        svm_acc_grow = svm_avg
    svm_kernel.append(svm_data)
exit()
for svm_data in svm_kernel:
    # Same as above but with standard deviation
    plt.figure()
    plt.title('Test Accuracy Avg vs Train size N for SVM with std {kernel}'.format(kernel=svm_data[0][3]))
    plt.xlabel('Train size N')
    plt.ylabel('Testing accuracy %')
    std = []
    std_uncertainty = []
    for point in svm_data:
        std.append(point[1])
        std_uncertainty.append(point[2])
    plt_min = round(min(std) - max(std_uncertainty), 1)
    plt_max = round(max(std) + max(std_uncertainty), 1)
    plt.grid(True)
    # Perceptron data in red
    plt.errorbar(
        [point[0] for point in svm_data],
        [point[1] for point in svm_data],
        [point[2] for point in svm_data],
        ecolor='r',
        marker='^'
    )


    plt.show()
