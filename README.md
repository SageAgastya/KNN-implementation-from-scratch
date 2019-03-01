# KNN-implementation-from-scratch

As we know KNN(K-Nearest Neighbours) is a classification technique which can be used for classification of binary classes. It is simple technique which only uses the euclidean distance to find the proximity of the the testing point with the classes involved.
The best reason for using this model lies in its simplicity in implementation and the fact that it doesn't require training.
Since the data need not to be trained, the cost of the time will be reduced.

NOTE:
1. Always set the value of k to an odd value. The reason is simple, if we choose even value of k(let's say k=2) then there is possibility that the euclidean distance of the testing point from its two closest points comes same, provided that the 2 closest neighbours belong to different classes, this will result in failure of KNN to predict the class of unknown point.

2. Don't use this model if the length of the data set is very large because higher number of data scatter points will introduce 
the situation discussed in 1st point described above. 

3. Another reason for not using this model when size of data is very large contributes to the laziness of the KNN-model.
