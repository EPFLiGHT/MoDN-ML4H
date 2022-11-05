# utility functions for the iio experiments
import math

import numpy as np
import torch


def split_qst(
    qst_obj, length_training, percentage_common_features=0.8, specific_data_size=0.2
):
    """split questionnaire object into centralized and local subsets

    Args:
        qst_obj: questionnaire object
        length_training: size of all training data
        percentage_common_features (float, optional): Percentage of centralized features. Defaults to 0.8.
        specific_data_size (float, optional): Percentage of local data size. Defaults to 0.2.

    Returns:
        _type_: _description_
    """
    common_features = list(
        np.random.choice(
            qst_obj.feature_names,
            size=math.ceil(percentage_common_features * len(qst_obj.feature_names)),
            replace=False,
        )
    )
    specific_features = [
        feature for feature in qst_obj.feature_names if feature not in common_features
    ]
    specific_indices = np.random.choice(
        qst_obj.training_data.indices,
        size=int(specific_data_size * length_training),
        replace=False,
    )
    common_indices = [
        elem for elem in qst_obj.training_data.indices if elem not in specific_indices
    ]
    qst_obj.specific_indices = specific_indices
    qst_obj.common_indices = common_indices
    print(f"common indices {len(qst_obj.common_indices)}")
    qst_obj.specific_features = specific_features
    print(f"specific indices {len(qst_obj.specific_indices)}")
    qst_obj.common_features = common_features
    qst_obj.common_features_train = [
        qst_obj.traininig_data_indices_mapping[index]
        for index in qst_obj.common_indices
    ]
    qst_obj.specific_features_train = [
        qst_obj.traininig_data_indices_mapping[index]
        for index in qst_obj.specific_indices
    ]
    return qst_obj


def delete_values(qst, X_cont_train, X_cat_train):
    """Simulate scenario in which some values of features are not available

    Args:
        qst: questionnaire object
        X_cont_train: training array of continuous features
        X_cat_train: training array of categorical features

    Returns: modified arrays
    """
    new_train_feature_group = {}
    for level in qst.train_group_order_features.keys():
        new_train_feature_group[level] = {}
        for feature_group in qst.train_group_order_features[level]:
            if len(set(qst.specific_features).intersection(list(feature_group))) > 0:
                common_feature_group = tuple(
                    sorted(list(set(list(feature_group)) - set(qst.specific_features)))
                )
                new_train_feature_group[level][common_feature_group] = [
                    elem
                    for elem in qst.train_group_order_features[level][feature_group]
                    if elem in qst.common_features_train
                ]
                new_train_feature_group[level][feature_group] = [
                    elem
                    for elem in qst.train_group_order_features[level][feature_group]
                    if elem not in qst.common_features_train
                ]
            else:
                new_train_feature_group[level][
                    feature_group
                ] = qst.train_group_order_features[level][feature_group]
    qst.train_group_order_features = new_train_feature_group
    for feature in qst.specific_features:
        for patient in qst.common_indices:
            train_index = qst.traininig_data_indices_mapping[patient]
            index = qst.feature_names.index(feature)
            qst.raw_data.loc[patient, feature] = np.nan
            qst.questionnaire_data[patient, index] = np.nan
            if feature in qst.continuous_features_names:
                index = qst.continuous_features_names.index(feature)
                X_cont_train[train_index, index] = np.nan
            else:
                index = qst.categorical_features_names.index(feature)
                X_cat_train[train_index, index] = np.nan

    return X_cont_train, X_cat_train


def normalize(qst, X_train, X_valid, X_test):
    """
    normalize train, validation and test datasets using mean and variance
    of train dataset
    :param qst: questionnaire dataset object
    :param X_train: training data
    :param X_valid: validation data
    :param X_test: test data
    :return: normalized train, validation and test sets
    """

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


def preprocess_data(qst, size_test, size_valid):
    """preprocess data for distributed training

    Args:
        qst: questionnaire object
        size_test: proportion for test
        size_valid: proportion for valid

    Returns: train/test/valid tensors
    """
    data_size = len(qst)
    # get train, valid and test patient indices
    testing_indices = np.random.choice(
        range(data_size), size=math.ceil(size_test * data_size), replace=False
    )
    training_indices = np.array(
        [index for index in range(data_size) if index not in testing_indices]
    )
    validation_indices = np.random.choice(
        training_indices,
        size=math.ceil(len(training_indices) * size_valid),
        replace=False,
    )
    training_indices = np.array(
        [elem for elem in training_indices if elem not in validation_indices]
    )

    training_data, validation_data, testing_data = (
        torch.utils.data.Subset(qst, training_indices),
        torch.utils.data.Subset(qst, validation_indices),
        torch.utils.data.Subset(qst, testing_indices),
    )
    qst.training_data = training_data
    qst.testing_data = testing_data
    qst.validation_data = validation_data

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

    qst.remap_test_valid_set()

    y_train, y_valid, y_test = (
        qst.labels[training_data.indices],
        qst.labels[validation_data.indices],
        qst.labels[testing_data.indices],
    )

    (
        X_cont_train,
        X_cont_valid,
        X_cont_test,
        mean_cont,
        std_cont,
        min_cont,
        max_cont,
    ) = normalize(qst, X_cont_train, X_cont_valid, X_cont_test)

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
        X_cont_train,
        X_cont_valid,
        X_cont_test,
        X_cat_train,
        X_cat_valid,
        X_cat_test,
        y_train,
        y_valid,
        y_test,
    )
