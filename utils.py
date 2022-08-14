import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, encoder="onehot", normalizer="minmax"):
        self.feature_arrays = []
        self.feature_names = []
        self.label_array = []
        self.enc_dict = {}
        self.encoder_class = OneHotEncoder if encoder == "onehot" else OrdinalEncoder
        self.normalizer_class = MinMaxScaler if normalizer == "minmax" else StandardScaler

    def add_feature(self, feature_array, name):
        self.feature_arrays.append(feature_array)
        for i in range(feature_array.shape[1]):
            self.feature_names.append(name + str(i))
    
    def add_label(self, label_array):
        if len(label_array.shape) == 1:
            label_array = label_array.reshape(-1, 1)
        else:
            self.label_array = label_array

    def get_data(self):
        return np.hstack(self.feature_arrays), self.label_array.flatten(), self.feature_names

    def reset_for_test(self):
        self.feature_arrays = []
        self.feature_names = []
        self.label_array = []


    def encode(self, df, column_name, mode="train", top_n=None):
        if mode == "train": 
            encoder = self.encoder_class(sparse=False)
            encoder = encoder.fit(df[column_name].values.reshape(-1, 1))
            self.enc_dict[column_name] = (top_n, encoder)
        else:
            top_n, encoder = self.enc_dict[column_name]

        if top_n:
            return encoder.transform(df[column_name].values.reshape(-1, 1))[:, :top_n]
        else:
            return encoder.transform(df[column_name].values.reshape(-1, 1))

    def normalize(self, df, column_name, mode="train"):
        if mode == "train":
            normalizer = self.normalizer_class()
            normalizer = normalizer.fit(df[column_name].values.reshape(-1, 1))
            self.enc_dict[column_name] = normalizer
        else:
            normalizer = self.enc_dict[column_name]
        return normalizer.transform(df[column_name].values.reshape(-1, 1))



def one_vs_all(y, target=0):
    if target == None:
        return y
    y_ = y.copy()
    y_[y!=target] = 0
    y_[y==target] = 1
    return y_

def plot_roc(model, X, y, model_name="model"):
    y_pred_probs, _ = get_prob_and_pred(model, X)
    fpr, tpr, _ = roc_curve(one_vs_all(y), y_pred_probs, drop_intermediate=False)
    auc = roc_auc_score(y, y_pred_probs)
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr)
    plt.title(f"ROC curve for {model_name}")
    plt.xlabel("FP")
    plt.ylabel("TP")
    plt.fill_between(fpr, tpr, alpha=0.6)
    plt.text(x=.5, y=.5, s=f"{auc=:.2f}", fontsize=20)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

def get_prob_and_pred(model, X, threshold=0.5):
    y_pred_probs = model.predict_proba(X)
    y_pred_probs = y_pred_probs[:, 1:].sum(axis=-1)
    y_pred = (y_pred_probs>threshold)*1.

    return y_pred_probs, y_pred