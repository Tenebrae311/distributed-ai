import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import shap

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from civiclink.utils import get_root_dir

def main(test_samples):
    # load data
    root_dir = get_root_dir()
    fpath = root_dir/"data/fake_steuerdaten_labels_not_random.csv"
    df = pd.read_csv(fpath, delimiter=";")

    X,y = df.drop(columns="Label"), df["Label"]

    for col in ["Familienstand", "Steuerklasse", "Bundesland","Religionszugehörigkeit", 
                "Einkunftsart","Branche_Selbstaendig"]:
        if col in X.columns:
            X[col] = X[col].astype("category")

    #X["Summe_Einkuenfte_Brutto"].astype("float")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 1, shuffle = True)

    bst = XGBClassifier(enable_categorical=True)

    bst.fit(X_train, y_train)
    explainer = shap.TreeExplainer(bst)

    X_pos = test_samples
    X_sample = X_pos.iloc[0:1]
    shap_values  = explainer.shap_values(X_sample)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    shape_values_1_sample = mean_abs_shap
    shape_values_1_sample = np.sort(shape_values_1_sample)[::-1]
    total = shape_values_1_sample.sum()

    # get 90 %
    sum = 0.0
    top_90 = 0
    for i, feat in enumerate(shape_values_1_sample):
        sum += feat
        if sum/total > 0.8:
            top_90 = i
            break
    top_idx = np.argsort(mean_abs_shap)[-top_90:]  # largest 13
    X_top = X_sample.iloc[:, top_idx]
    shap_values_top = shap_values[:, top_idx]
    shap.summary_plot(shap_values_top, X_top)

    fig = plt.gcf()

    return fig
