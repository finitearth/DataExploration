import numpy as np
from sklearn.preprocessing import OneHotEncoder

class Dataset:
    def __init__(self):
        self.feature_arrays = []
        self.feature_names = []
        self.label_array = []

    def add_feature(self, feature_array, name):
        self.feature_arrays.append(feature_array)
        for i in range(feature_array.shape[1]):
            self.feature_names.append(name + str(i))
    
    def add_label(self, label_array):
        if len(label_array.shape) == 1:
            label_array = label_array.reshape(-1, 1)
        else:
            self.label_array = label_array

    def get_X(self):
        return np.hstack(self.feature_arrays)
    
    def get_y(self):
        return self.label_array.flatten()

    def get_feature_names(self):
        return self.feature_names

def get_one_hot(df, column_name, top_n=None):
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    if top_n:
        return encoder.fit_transform(df[column_name].values.reshape(-1, 1))[:, :top_n]
    else:
        return encoder.fit_transform(df[column_name].values.reshape(-1, 1))