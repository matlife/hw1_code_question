__author__ = 'matthewlei'
from run_knn import run_knn
from utils import *

train_data, train_labels = load_train()
valid_data, valid_target = load_valid()

k_list = [1, 3, 5, 7, 9]

for k in k_list:
    predicted_labels = run_knn(k, train_data, train_labels, valid_data)

    #Count correct labels according to validation set
    correct = 0
    for i in range(len(predicted_labels)):
        if predicted_labels[i] == valid_target[i]:
            correct += 1

    print("K value: " + str(k) + ", Number correct: " + str(correct) + " out of " + str(len(valid_target)))
    print("Accuracy: " + str(correct / float(len(valid_target))))

    # These correspond. Too little --> higher variance... makes it inaccurate too much --> more bias makes it inaccurate