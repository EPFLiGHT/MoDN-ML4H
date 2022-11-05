# utility functions

import random

import numpy as np
import pandas as pd

from modules import *


# functions for preprocessing data
def normalize(qst, X_train, X_valid, X_test, features):
    """
    normalize train, validation and test datasets using mean and variance
    of train dataset
    :param qst: questionnaire dataset object
    :param X_train: training data
    :param X_valid: validation data
    :param X_test: test data
    :param features: boolean, true if variables to normalize are features
    :return: normalized train, validation and test sets
    """

    if features:
        # remove columns containing only nans in X_cont_train from X_cont_train, X_cont_valid and X_cont_test
        features_to_remove = [
            index
            for index, feature in enumerate(np.all(np.isnan(X_train), axis=0))
            if feature
        ]
        # shift indices in decision tree structure
        for feature in features_to_remove:
            for key in qst.tree_struct.keys():
                qst.tree_struct[key] = [
                    elem if elem < feature else elem - 1
                    for elem in qst.tree_struct[key]
                ]
        X_valid = X_valid[:, ~np.all(np.isnan(X_train), axis=0)]
        X_test = X_test[:, ~np.all(np.isnan(X_train), axis=0)]
        X_train = X_train[:, ~np.all(np.isnan(X_train), axis=0)]
        # change qst.num_continuous_features if some columns have been dropped
        qst.num_continuous_features = X_train.shape[1]
    else:
        min_ = None
    min = np.nanmin(X_train, axis=0)
    max = np.nanmax(X_train, axis=0)
    mean = np.nanmean(X_train, axis=0)
    std = np.nanstd(X_train, axis=0)
    X_train[:, std > 0] = (X_train[:, std > 0] - mean[std > 0]) / std[std > 0]
    X_valid[:, std > 0] = (X_valid[:, std > 0] - mean[std > 0]) / std[std > 0]
    X_test[:, std > 0] = (X_test[:, std > 0] - mean[std > 0]) / std[std > 0]
    X_train[:, std == 0] = X_train[:, std == 0] - mean[std == 0]
    X_valid[:, std == 0] = X_valid[:, std == 0] - mean[std == 0]
    X_test[:, std == 0] = X_test[:, std == 0] - mean[std == 0]

    return X_train, X_valid, X_test, mean, std, min, max


def impute_mean(X_train, X_valid, X_test):
    """
    Replace np.nan in train, validation and test sets
    by mean of feature in X_cont_train
    :param X_train: training data
    :param X_valid: validation data
    :param X_test: test data
    :return:
    """
    # compute mean values for features of X_cont_train
    mean_values = np.nanmean(X_train, axis=0)
    # iterate over covariates
    for index in range(X_train.shape[1]):
        X_train[:, index][np.isnan(X_train[:, index])] = mean_values[index]
        X_test[:, index][np.isnan(X_test[:, index])] = mean_values[index]
        X_valid[:, index][np.isnan(X_valid[:, index])] = mean_values[index]
    return


def preprocess_data_epoct(
    qst,
    valid_size=0.2,
    test_size=0.2,
    tensors=True,
    imput_strategy=None,
    fold=None,
):
    """
    Split questionnaire dataset into training, validation and test sets. Normalize the
    data and apply imputation strategy.
    :param qst: QuestionnaireDataset object
    :param valid_size: proportion of data to be used as validation set
    :param test_size: proportion of data to be used as test set
    :param tensors: boolean, whether to convert datasets to torch tensors
    :param imput_strategy: imputation strategy to be used to replace np.nans in features
    :param fold: precomputed fold inices
    :return: preprocessed X_cont_train, X_cont_valid, X_cont_test, y_train, y_valid, y_test
    """
    # get train, valid and test patient indices

    training_data, validation_data, testing_data = qst.ml_dataset(
        valid_size, test_size, fold
    )

    if tensors:

        X_cont_train, X_cont_valid, X_cont_test = (
            qst.continuous_features[training_data.indices],
            qst.continuous_features[validation_data.indices],
            qst.continuous_features[testing_data.indices],
        )
        X_cat_train, X_cat_valid, X_cat_test = (
            qst.categorical_features[training_data.indices],
            qst.categorical_features[validation_data.indices],
            qst.categorical_features[testing_data.indices],
        )

        X_train, X_valid, X_test = (
            qst.questionnaire_data[training_data.indices],
            qst.questionnaire_data[validation_data.indices],
            qst.questionnaire_data[testing_data.indices],
        )
        qst.remap_test_valid_set()
    else:
        X_train, X_valid, X_test = (
            qst.questionnaire_data[training_data.indices],
            qst.questionnaire_data[validation_data.indices],
            qst.questionnaire_data[testing_data.indices],
        )
        (
            X_cont_train,
            X_cont_valid,
            X_cont_test,
            X_cat_train,
            X_cat_valid,
            X_cat_test,
        ) = (None, None, None, None, None, None)

    y_train, y_valid, y_test = (
        qst.labels[training_data.indices],
        qst.labels[validation_data.indices],
        qst.labels[testing_data.indices],
    )
    if tensors:

        (
            X_cont_train,
            X_cont_valid,
            X_cont_test,
            mean_cont,
            std_cont,
            min_cont,
            max_cont,
        ) = normalize(qst, X_cont_train, X_cont_valid, X_cont_test, features=True)
        X_train, X_valid, X_test, _, _, _, _ = normalize(
            qst, X_train, X_valid, X_test, features=False
        )

    else:
        X_train, X_valid, X_test, mean_cont, std_cont, min_cont, max_cont = normalize(
            qst, X_train, X_valid, X_test, features=True
        )
    if imput_strategy == "mean":
        impute_mean(X_train, X_valid, X_test)
    if tensors:
        X_cont_train = torch.from_numpy(X_cont_train.astype(np.float32))
        X_cont_valid = torch.from_numpy(X_cont_valid.astype(np.float32))
        X_cont_test = torch.from_numpy(X_cont_test.astype(np.float32))

        X_cat_train = torch.from_numpy(X_cat_train.astype(np.float32))
        X_cat_valid = torch.from_numpy(X_cat_valid.astype(np.float32))
        X_cat_test = torch.from_numpy(X_cat_test.astype(np.float32))
        y_train, y_valid, y_test = (
            torch.LongTensor(y_train),
            torch.LongTensor(y_valid),
            torch.LongTensor(y_test),
        )

    return (
        X_train,
        X_valid,
        X_test,
        X_cont_train,
        X_cont_valid,
        X_cont_test,
        X_cat_train,
        X_cat_valid,
        X_cat_test,
        y_train,
        y_valid,
        y_test,
        mean_cont,
        std_cont,
        min_cont,
        max_cont,
    )


# functions to instantiate modules


def get_binary_disease_decoders(qst, STATE_SIZE=4):
    """
    Instantiate categorical binary (yes/no) decoder for each disease
    :param qst: questionnaire data set object
    :param STATE_SIZE: size of intermediate state
    :return: decoder modules for each disease and list of parameters to train
    """

    binary_decoders = {
        name: EpoctBinaryDecoder(STATE_SIZE) for name in qst.disease_names
    }

    return binary_decoders


def get_feature_decoders(qst, STATE_SIZE):
    """
    Instantiate decoders for features
    :param qst: questionnaire data set object
    :param STATE_SIZE: size of intermediate hidden state
    :return: decoder for each module and list of parameters to train
    """

    continuous_feature_decoders = {
        name: EpoctDistributionDecoder(STATE_SIZE)
        for name in qst.continuous_features_names
    }

    categorical_feature_decoders = {
        name: EpoctCategoricalDecoder(STATE_SIZE, qst.categorical_features_dict[name])
        for name in qst.categorical_features_names
    }

    return continuous_feature_decoders, categorical_feature_decoders


def get_feature_encoders(qst, STATE_SIZE):
    """
    Instantiate encoder module for each feature
    :param qst: questionnaire data set object
    :param STATE_SIZE: size of intermediate state
    :return: encoder module for each feature and list of parameters to train
    """

    # dictionnary of encoders
    encoders = {name: EpoctEncoder(STATE_SIZE) for name in qst.feature_names}

    return encoders


###### loss functions ########
def mse(a, b):
    """
    compute mean squared error of between two tensors
    :param a: first tensor
    :param b: second tensor
    :return: mse
    """
    return torch.mean((a - b) ** 2)


def neg_log_likelihood_1D(y, mu, log_sigma):
    return (
        torch.sum(log_sigma + (y - mu) ** 2 / (2 * torch.exp(log_sigma) ** 2))
        / y.shape[0]
    )


####### other ###############
def get_non_nan_features(X, continuous=True):
    """
    for each feature of X gets values and indices of non nan values
    :param X: tensor of features
    :param continuous: boolean, whether feature is continuous or categorical
    :return: two lists of tensors containing non nan features and indices of patients
    with non nan features
    """
    if X is None:
        return None, None
    non_nan_features = []
    non_nan_patients = []
    for index in range(X.shape[1]):
        # select only non nan values for that feature
        if continuous:
            non_nan_features.append(
                torch.masked_select(X[:, index], ~torch.isnan(X[:, index])).view(-1, 1)
            )
        else:
            non_nan_features.append(
                torch.masked_select(X[:, index], ~torch.isnan(X[:, index]))
                .long()
                .view(-1, 1)
            )
        # select all patients with non nan value for that feature
        non_nan_patients.append(
            torch.masked_select(
                torch.tensor(range(len(X))), ~torch.isnan(X[:, index])
            ).numpy()
        )

    return non_nan_features, non_nan_patients


def one_hot_encoding(num_classes, array):
    """
    Transform vector of targets with num_classes categories into num_classes target vectors
    with one hot encoding
    :param num_classes: number of classes in array
    :param array: target vector
    :return: array of size len(vector) * cum_classes containing the one hot encoded representation of array
    """
    b = torch.zeros(len(array), num_classes)
    for patient in range(len(b)):
        if np.isnan(array[patient]):
            b[patient, :] = np.nan
        else:
            b[patient, int(array[patient])] = 1
    return b


def shuffle(array):
    array = list(array)
    return random.sample(array, len(array))


# metrics
def compute_metrics(y_pred, y_true, class_to_compare=None):
    """
    preprocess vectors to be able to compute the metrics
    :param y_pred: predictted targets
    :param y_true: true targets
    :param class_to_compare: if vectors are not one hot encoded, compute only the metrics wrt the class=class_to_compare
    :return:
    """
    if class_to_compare is not None:
        # 1 if patient has disease = class_to_compare else 0
        y_pred_modified = [
            0 if y_pred[index] != class_to_compare else 1
            for index in range(len(y_pred))
        ]
        y_true_modified = [
            0 if y_true[index] != class_to_compare else 1
            for index in range(len(y_true))
        ]
        (
            precision_0,
            recall_0,
            F1_0,
            precision_1,
            recall_1,
            F1_1,
        ) = precision_recall_f1(y_pred_modified, y_true_modified)
    else:
        (
            precision_0,
            recall_0,
            F1_0,
            precision_1,
            recall_1,
            F1_1,
        ) = precision_recall_f1(y_pred, y_true)

    return (
        np.around(precision_0, decimals=2),
        np.around(recall_0, decimals=2),
        np.around(F1_0, decimals=2),
        np.around(precision_1, decimals=2),
        np.around(recall_1, decimals=2),
        np.around(F1_1, decimals=2),
        np.around((F1_0 + F1_1) / 2, decimals=2),
    )


def precision_recall_f1(y_pred, y_true):
    """
    compute precision, recall and f_1 scores for both classes
    :param y_pred: predictions
    :param y_true: true values
    :return: precision_0, recall_0, F1_0, precision_1, recall_1, F1_1
    """
    TN = len(
        [
            index
            for index in range(len(y_pred))
            if (y_pred[index] == 0 and y_true[index] == 0)
        ]
    )
    TP = len(
        [
            index
            for index in range(len(y_pred))
            if (y_pred[index] == 1 and y_true[index] == 1)
        ]
    )
    FN = len(
        [
            index
            for index in range(len(y_pred))
            if (y_pred[index] == 0 and y_true[index] == 1)
        ]
    )
    FP = len(
        [
            index
            for index in range(len(y_pred))
            if (y_pred[index] == 1 and y_true[index] == 0)
        ]
    )
    # class 0
    if TN + FN == 0:
        precision_0 = 0
    else:
        precision_0 = (TN) / (TN + FN)
    if TN + FP == 0:
        recall_0 = 0
    else:
        recall_0 = (TN) / (TN + FP)
    if (precision_0 + recall_0) == 0:
        F1_0 = 0
    else:
        F1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0)
    # class 1
    if TP + FP == 0:
        precision_1 = 0
    else:
        precision_1 = (TP) / (TP + FP)
    if TP + FN == 0:
        recall_1 = 0
    else:
        recall_1 = (TP) / (TP + FN)
    if (precision_1 + recall_1) == 0:
        F1_1 = 0
    else:
        F1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1)

    return (precision_0, recall_0, F1_0, precision_1, recall_1, F1_1)


def conf_matrix(y_pred, y_true):
    """confusion matrix"""
    df = pd.DataFrame(
        {"Target": y_true, "Prediction": y_pred}, columns=["Target", "Prediction"]
    )
    confusion_matrix = pd.crosstab(
        df["Target"], df["Prediction"], rownames=["Target"], colnames=["Prediction"]
    )
    return confusion_matrix


def latex_table_from_df(df):
    print(df.to_latex(index=False))
    return


# computations


def calibration(qst, disease, level, decoder_output, true_targets, num_intervals):
    """

    :param qst:
    :param disease: name of disease. If "all" calibration plot wrt all the diseases at once
    :param level: level in the tree
    :param decoder_output: tensor of shape (max_tree_depth, num_patients, num_diseases)
    containing the probabilities at each level in the tree for each patient for each disease
    :param true_targets: tensor of shape (num_patients, num_diseases) where true_targets(i,j) =1
    if patient i has disease j and 0 otherwise
    :param num_intervals: number of bins for calibration computation
    :return:
    """
    if disease == "all":
        # expected calibration error
        ECE = 0
        accuracies = np.empty((len(qst.disease_names), num_intervals))
        confidences = np.empty((len(qst.disease_names), num_intervals))
        obs_per_bin = np.empty((len(qst.disease_names), num_intervals))
        for index in range(len(qst.disease_names)):
            probabilities = decoder_output[level, :, index]
            for m in range(1, num_intervals + 1):
                # set of patients with predicted probabilities s.t.
                # (m-1)/num_intervals < probabilities[patient] <= m/num_intervals
                B_m = set()
                for patient in range(len(true_targets)):
                    if m == 1 and probabilities[patient] == 0:
                        B_m.add(patient)
                    if (
                        (m - 1) / num_intervals
                        < probabilities[patient]
                        <= m / num_intervals
                    ):
                        B_m.add(patient)
                # accuracy of B_m
                A_m = 0
                # confidence of B_m
                C_m = 0
                obs_per_bin[index, m - 1] = len(B_m)
                for elem in B_m:
                    if true_targets[elem, index] == 1:
                        A_m += 1 / len(B_m)
                    C_m += 1 / len(B_m) * probabilities[elem].item()
                if len(B_m) == 0:
                    A_m = np.nan
                    C_m = np.nan
                accuracies[index, m - 1] = A_m
                confidences[index, m - 1] = C_m

        accuracies = np.mean(accuracies, axis=0, where=~np.isnan(accuracies))
        confidences = np.mean(confidences, axis=0, where=~np.isnan(confidences))
        obs_per_bin = np.sum(obs_per_bin, axis=0)
        for m in range(num_intervals):
            if obs_per_bin[m] != 0:
                ECE += (
                    obs_per_bin[m]
                    / (len(true_targets) * len(qst.disease_names))
                    * abs(accuracies[m] - confidences[m])
                )
        # if no observations in interval (for plotting purposes)
        for m in range(1, num_intervals + 1):
            if obs_per_bin[m - 1] == 0:
                confidences[m - 1] = (2 * m - 1) / (2 * num_intervals)
    else:
        # expected calibration error
        ECE = 0
        index = qst.disease_names.index(disease)
        probabilities = decoder_output[level, :, index]
        accuracies = []
        confidences = []
        # number of observations per bin
        obs_per_bin = []
        for m in range(1, num_intervals + 1):
            # set of patients with predicted probabilities s.t.
            # (m-1)/num_intervals < probabilities[patient] <= m/num_intervals
            B_m = set()
            for patient in range(len(true_targets)):
                if m == 1 and probabilities[patient] == 0:
                    B_m.add(patient)
                if (
                    (m - 1) / num_intervals
                    < probabilities[patient]
                    <= m / num_intervals
                ):
                    B_m.add(patient)
            # accuracy of B_m
            A_m = 0
            # confidence of B_m
            C_m = 0
            obs_per_bin.append(len(B_m))
            if len(B_m) == 0:
                C_m = (2 * m - 1) / (2 * num_intervals)
            else:
                for elem in B_m:
                    if true_targets[elem, index] == 1:
                        A_m += 1 / len(B_m)
                    C_m += 1 / len(B_m) * probabilities[elem].item()
                ECE += len(B_m) / len(true_targets) * abs(A_m - C_m)
            accuracies.append(A_m)
            confidences.append(C_m)
        # print(f'\n level {level} disease {disease} acc {accuracies} conf{confidences}')

    return (
        np.round(np.array(accuracies), 2),
        np.round(np.array(confidences), 2),
        np.array(obs_per_bin),
        np.round(ECE, 2),
    )


if __name__ == "__main__":
    print(f"\n End of script")
