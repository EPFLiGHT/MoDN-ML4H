# script to run all the IIO experiments (i.e. centralized training + fine-tuning/compartmentalisation).
# The trained centralized models are saved in the updated_centralized file
# and are then loaded for the fine-tuning/compartmentalisation


import os
import random

import numpy as np
import pandas as pd
import torch

import distributed_training_parameters
from dataset_generation import EPOCTQuestionnaireDataset
from training_procedures_epoct import train_and_test_modules_distributed
from utils_distributed import *
from utils_epoct import get_targets, process_features

print(os.getcwd())
# load complete unlabeled data

os.chdir("data")
print(os.getcwd())
path = os.path.join(os.getcwd(), "data.csv")
print(path)
data = pd.read_csv(path)
print(os.getcwd())
os.chdir("../models")

# select features and targets
df, categorical_features_dict = process_features(data)
features, targets, categorical_features_dict = get_targets(
    df, categorical_features_dict
)

num_classes = targets.shape[1]
possible_target_values = targets.columns
# instantiate questionnaire data set object
qst_obj = EPOCTQuestionnaireDataset(
    features, targets, num_classes, possible_target_values, categorical_features_dict
)
# lower baseline
f1_macro = np.empty(5)
for s in range(0, 5):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

    # split into train and test
    (
        X_cont_train,
        X_cont_valid,
        X_cont_test,
        X_cat_train,
        X_cat_valid,
        X_cat_test,
        y_train,
        y_valid,
        y_test,
    ) = preprocess_data(qst_obj, size_test=0.1, size_valid=0.1)
    print(f"size of test set {len(y_test)}")
    qst_obj = split_qst(
        qst_obj,
        len(X_cont_train),
        percentage_common_features=0.8,
        specific_data_size=0.2,
    )
    X_cont_train, X_cat_train = delete_values(qst_obj, X_cont_train, X_cat_train)
    # common
    lr_feature_decoders = {
        feature_name: 1e-3
        if feature_name in qst_obj.categorical_features_names
        else 1e-4
        for feature_name in qst_obj.feature_names
    }
    lr_encoders = {feature_name: 1e-2 for feature_name in qst_obj.feature_names}

    training_parameters = distributed_training_parameters.param["lower-baseline"]
    qst_obj.training_parameters = training_parameters
    f1_macro[s] = train_and_test_modules_distributed(
        qst_obj,
        X_cont_train,
        X_cont_valid,
        X_cont_test,
        X_cat_train,
        X_cat_valid,
        X_cat_test,
        y_train,
        y_valid,
        y_test,
        method="lower-baseline",
        seed=s,
    )
print(f1_macro)
print(f"mean : {np.mean(f1_macro)}")
np.save("updated_centralized/lower-baseline", f1_macro)
# other models
for method in ["common", "compartm", "fine-tune", "upper-baseline"]:
    print(f"method {method}")
    f1_macro = np.empty(5)
    for perc in [60, 80, 90, 100]:
        for s in range(0, 5):
            random.seed(s)
            np.random.seed(s)
            torch.manual_seed(s)

            # split into train and test
            (
                X_cont_train,
                X_cont_valid,
                X_cont_test,
                X_cat_train,
                X_cat_valid,
                X_cat_test,
                y_train,
                y_valid,
                y_test,
            ) = preprocess_data(qst_obj, size_test=0.1, size_valid=0.1)
            print(f"size of test set {len(y_test)}")
            qst_obj = split_qst(
                qst_obj,
                len(X_cont_train),
                percentage_common_features=perc / 100,
                specific_data_size=0.2,
            )
            X_cont_train, X_cat_train = delete_values(
                qst_obj, X_cont_train, X_cat_train
            )
            # common
            lr_feature_decoders = {
                feature_name: 1e-3
                if feature_name in qst_obj.categorical_features_names
                else 1e-4
                for feature_name in qst_obj.feature_names
            }
            lr_encoders = {feature_name: 1e-2 for feature_name in qst_obj.feature_names}
            training_parameters = distributed_training_parameters.param[method][perc]
            qst_obj.training_parameters = training_parameters
            f1_macro[s] = train_and_test_modules_distributed(
                qst_obj,
                X_cont_train,
                X_cont_valid,
                X_cont_test,
                X_cat_train,
                X_cat_valid,
                X_cat_test,
                y_train,
                y_valid,
                y_test,
                percentage=perc,
                method=method,
                seed=s,
            )
        print(f1_macro)
        print(f"mean : {np.mean(f1_macro)}")
        np.save("updated_centralized/" + str(method) + "_" + str(perc), f1_macro)

print("End of script")
