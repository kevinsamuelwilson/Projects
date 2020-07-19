# script that takes data and loads in into pickle form
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import dok_matrix, coo_matrix

NUM_CLASSES = 3993
NUM_FEATURES = 5000

df = pd.read_csv('data/train.csv')

n = len(df) # num rows in dataset

# create class matrix

# initialize empty sparse matrix
# using dok (dictionary of keys) sparse matrix to initialize,
# then will tranfer it into a coo (coordinate) form sparse matrix
sparse_class_mat = dok_matrix((n, NUM_CLASSES + 1), dtype = np.bool)

for i in range(n):
    labels_list = df.iloc[i,1].split(',')
    for label in labels_list:
        try:
            sparse_class_mat[i, int(label)] = True
        except ValueError:

            # some rows (see row 93 for example) don't have data in the
            # label column and instead has the first value of the feature
            # column.
            # So I created a new label at the end (at index NUM_CLASSES) for these
            # entries and set that value to 1 (it is like a 'no label given' label)
            # Then, I add the feature that was put in the label column into the feature
            # column
            sparse_class_mat[i,NUM_CLASSES] = True
            features = df.iloc[i,2]
            df.iloc[i,1] = str(NUM_CLASSES)
            df.iloc[i,2] = label + ' ' + features

# turn into COO matrix
sparse_class_mat = sparse_class_mat.asformat('coo')

f = open('data/train_labels_sparse.pickle', 'wb')
pickle.dump(sparse_class_mat, f)
f.close()


# create label matrix

# initialize empty sparse matrix
sparse_feature_mat = dok_matrix((n, NUM_FEATURES), dtype = np.float32)

for i in range(n):
    feature_list = df.iloc[i, 2].split(' ')
    for feature in feature_list:
        key, value = feature.split(':')
        sparse_feature_mat[i, int(key)] = np.float(value)

sparse_feature_mat = sparse_feature_mat.asformat('coo')

f = open('data/train_features_sparse.pickle', 'wb')
pickle.dump(sparse_feature_mat, f)
f.close()

# ----------------------------------------------------------------------------
# copying same code but for dev data

df = pd.read_csv('data/dev.csv')

n = len(df) # num rows in dataset

# create class matrix

# initialize empty sparse matrix
# using dok (dictionary of keys) sparse matrix to initialize,
# then will tranfer it into a coo (coordinate) form sparse matrix
sparse_class_mat = dok_matrix((n, NUM_CLASSES + 1), dtype = np.bool)

for i in range(n):
    labels_list = df.iloc[i,1].split(',')
    for label in labels_list:
        try:
            sparse_class_mat[i, int(label)] = True
        except ValueError:

            # some rows (see row 93 for example) don't have data in the
            # label column and instead has the first value of the feature
            # column.
            # So I created a new label at the end (at index NUM_CLASSES) for these
            # entries and set that value to 1 (it is like a 'no label given' label)
            # Then, I add the feature that was put in the label column into the feature
            # column
            sparse_class_mat[i,NUM_CLASSES] = True
            features = df.iloc[i,2]
            df.iloc[i,1] = str(NUM_CLASSES)
            df.iloc[i,2] = label + ' ' + features

# turn into COO matrix
sparse_class_mat = sparse_class_mat.asformat('coo')

f = open('data/dev_labels_sparse.pickle', 'wb')
pickle.dump(sparse_class_mat, f)
f.close()


# create label matrix

# initialize empty sparse matrix
sparse_feature_mat = dok_matrix((n, NUM_FEATURES), dtype = np.float32)

for i in range(n):
    feature_list = df.iloc[i, 2].split(' ')
    for feature in feature_list:
        key, value = feature.split(':')
        sparse_feature_mat[i, int(key)] = np.float(value)

sparse_feature_mat = sparse_feature_mat.asformat('coo')

f = open('data/dev_features_sparse.pickle', 'wb')
pickle.dump(sparse_feature_mat, f)
f.close()