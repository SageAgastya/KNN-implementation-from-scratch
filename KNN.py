import math
import numpy as np
import pandas as pd
import random


# working function-> KNN
# predict->data point to be predicted
#k->count of closest neighbours
def KNN(data_set, predict, k):
    if (k <= len(data_set)):
        print "impossible"
        return
    else:
        distance =[]          #contains euclidean distance with the class to which data point belong to
        vote =[]              #contains only the k-closest groups in sorted order


        for group in data_set:                 #checking euclidean distance
            for features in data_set[group]:
                euclid_dist = math.sqrt(sum((np.array(features) - np.array(predict)) ** 2))
                distance.append([euclid_dist, group])

        distance.sort()
        for value in range(k):
            vote.append(distance[value][1])

        max_occur = max(vote, key=vote.count)        #count of closest neighbour
        res=max(vote)                                #predicted class
        total = k
        confidence = float(max_occur / total)
        return max_occur, confidence, res


df = pd.read_csv("/home/rishabh/Desktop/rishabh/b/machine learning/svm_proj-prac/afmls.csv")
df.replace("?", -99999, inplace=True)
df.drop(["id", "r"], 1, inplace=True)
full_data = np.array(df, dtype=np.float64)
full_data = full_data.tolist()
random.shuffle(full_data)

test_per = 0.2
test_data = full_data[-int(test_per * len(full_data)):]
train_data = full_data[:-int(test_per * len(full_data))]
train_set = {2.0: [], 4.0: []}
test_set = {2.0: [], 4.0: []}

for train_features in train_data:             #data set manipulation
    if train_features[-1] == 2.0:
        train_set[2.0].append(train_features[:-1])
    if train_features[-1] == 4.0:
        train_set[4.0].append(train_features[:-1])

for test_features in test_data:                   #data set manipulation
    if test_features[-1] == 2.0:
        test_set[2.0].append(test_features[:-1])
    if test_features[-1] == 4.0:
        test_set[4.0].append(test_features[:-1])

cnt_correct = 0  # for counting the counts for which our prediction is correct
cnt_features = 0  # for counting the total features in the data set

for group in test_set:
    for i in test_set[group]:
        predicted_class, confidence,out = KNN(train_set, i, 5)  # predict class is predicted class of i..
        if predicted_class == group:
            cnt_correct += 1.0
        cnt_features += 1.0
        print str(i) + ":" + str(predicted_class)
    accuracy = cnt_correct / cnt_features



print confidence, accuracy



