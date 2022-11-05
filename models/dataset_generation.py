# classes to create and handle Questionnaire objects from the data
import copy
import math
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split

np.random.seed(0)


# classes


class QuestionnaireDataset(Dataset):
    """
    Questionnaire base class to handle the questionnaire-type datasets
    """

    def __init__(self, questionnaire_data, labels, num_classes, possible_target_values):
        """_summary_

        Args:
            questionnaire_data: raw questionnaire data (dataframe like)
            labels: target values (series like)
            num_classes: number of possible different target values (integer)
            possible_target_values: possible values of targets (list like)
        """
        super().__init__()
        # questionnaire data set
        self.questionnaire_data = questionnaire_data
        # targets
        self.labels = labels
        self.target_variable = {
            "name": "disease",
            "type": "ordinal",
            "num_classes": num_classes,
            "possible_values": possible_target_values,
            "value_count": self.number_labels_per_leafnode(),
        }
        self.training_data = None
        self.testing_data = None
        self.validation_data = None

    def __len__(self):
        return len(self.questionnaire_data)

    def __getitem__(self, patient_index):
        """
        Return a traditional / flat representation of the datapoint
        np.nan for missing data
        """

        return self.questionnaire_data[patient_index, :], self.labels[patient_index]

    def number_labels_per_leafnode(self):
        """

        :return: dict with possible labels as keys and number of data points with that label
        as value
        """
        return {
            label: len(self.labels[label][self.labels[label] == 1])
            for label in self.labels.columns
        }

    def ml_dataset(self, valid_size=0.2, test_size=0.2, fold=None):
        """
        Split, standardise and handle missing values of train, validation and test sets.
        :param valid_size: proportion of data to be used as validation set
        :param test_size: proportion of data to be used as test set

        """
        # split into train, validation and test
        if fold is None:
            training_data, validation_data, testing_data = random_split(
                self,
                [
                    len(self)
                    - math.ceil(test_size * len(self))
                    - math.ceil(valid_size * len(self)),
                    math.ceil(valid_size * len(self)),
                    math.ceil(test_size * len(self)),
                ],
                generator=torch.Generator().manual_seed(0),
            )
        else:
            training_indices, testing_indices = fold
            validation_indices = np.random.choice(
                training_indices,
                size=math.ceil(len(training_indices) / 5),
                replace=False,
            )
            training_indices = np.array(
                [elem for elem in training_indices if elem not in validation_indices]
            )
            training_data, validation_data, testing_data = (
                torch.utils.data.Subset(self, training_indices),
                torch.utils.data.Subset(self, validation_indices),
                torch.utils.data.Subset(self, testing_indices),
            )

        indices_to_keep = np.random.choice(
            training_data.indices,
            size=int(len(training_data.indices)),
            replace=False,
        )
        training_data.indices = indices_to_keep
        self.training_data = training_data
        self.testing_data = testing_data
        self.validation_data = validation_data
        return training_data, validation_data, testing_data

    def ml_2x5fold(self, times=5):
        """
        Split for k-fold cross validation
        :param times: number of folds
        """
        # 1. repeat 5 times
        # 2. shuffle indices
        # 3. split into 2 subsets and save both to be used as train/test
        fold_size = math.ceil(len(self) / 2)
        folds = []
        for num in range(times):
            shuffled_indices = random.sample(list(range(len(self))), len(self))
            folds.append(
                tuple([shuffled_indices[0:fold_size], shuffled_indices[fold_size:]])
            )

        return folds


class EPOCTQuestionnaireDataset(QuestionnaireDataset):
    """
    child of QuestionnaireDataset class, specific to e-POCT data set
    """

    def __init__(
        self,
        questionnaire_data,
        labels,
        num_classes,
        possible_target_values,
        categorical_features_dict,
    ):
        super().__init__(
            questionnaire_data, labels, num_classes, possible_target_values
        )
        self.raw_data = questionnaire_data.sort_index(axis=1)
        self.raw_labels = labels.sort_index(axis=1)
        self.feature_names = sorted(list(questionnaire_data.columns))
        self.questionnaire_data = questionnaire_data.sort_index(axis=1).to_numpy()
        self.labels = labels.sort_index(axis=1).to_numpy()
        self.disease_names = sorted(list(labels.columns))
        # dict containing categorical features as keys and number of different classes as values
        self.categorical_features_dict = categorical_features_dict
        self.continuous_features_names = sorted(
            list(
                set(questionnaire_data.columns) - set(categorical_features_dict.keys())
            )
        )
        self.categorical_features_names = sorted(list(categorical_features_dict.keys()))
        self.continuous_features = questionnaire_data[
            self.continuous_features_names
        ].to_numpy()
        self.categorical_features = questionnaire_data[
            self.categorical_features_names
        ].to_numpy()
        self.num_continuous_features = len(self.continuous_features_names)
        self.num_categorical_features = len(self.categorical_features_names)
        (
            self.complete_columns,
            self.dict_equivalences,
            self.dict_implications,
        ) = self.get_order_df()
        self.order_features = self.get_order_features()
        self.group_order_features = self.group_patients_by_order_features()
        # to store, train/valid/test indices from epoct and epoct plus data sets
        self.all_indices = None
        self.num_available_features = self.raw_data.count(axis=1)

    def delete_patient(self, indices):
        """
        delete all information and structure related to patient index
        :param indices: indices to drop
        """
        self.raw_data.drop(indices, inplace=True)
        self.raw_data.reset_index(inplace=True)
        mapping = dict(zip(list(self.raw_data["index"]), list(self.raw_data.index)))
        self.raw_data.drop(["index"], axis=1, inplace=True)
        self.raw_labels.drop(indices, inplace=True)
        self.raw_labels.reset_index(drop=True, inplace=True)
        self.questionnaire_data = np.delete(self.questionnaire_data, indices, axis=0)
        self.labels = np.delete(self.labels, indices, axis=0)
        if hasattr(self, "test_balises"):
            self.test_balises.drop(indices, inplace=True)
            self.test_balises.reset_index(drop=True, inplace=True)

        for level in self.group_order_features.keys():
            for feature_group in self.group_order_features[level]:
                new_list = []
                for elem in list(self.group_order_features[level][feature_group]):
                    if elem in indices:
                        pass
                    else:
                        # remap value
                        new_list.append(mapping[elem])
                self.group_order_features[level][feature_group] = new_list
        return

    def get_order_df(self):
        """
        get the equivalences and implications (i.e. which features come at the same time in
        the decision tree and which are the ones implying others) of the features of the data set
        :return: the features that are available for everyone (i.e. corresponding to triage questions,
        a dict with the features as keys and the equivalent features as values,
        a dict with the features as keys and the predecessors in the tree as values
        """
        n_patients = len(self)
        # get features available for all patients (triage questions)
        complete_columns = [
            column
            for column in self.raw_data.columns
            if len(self.raw_data[column].dropna()) == n_patients
        ]
        # for each feature, get set of patients with value for that feature
        df_features = pd.DataFrame({"feature": self.raw_data.columns})
        df_features["patient_set"] = [
            set(self.raw_data[feature].dropna().index)
            for feature in df_features["feature"]
        ]
        # dict of features such that if featureA is in dict[featureB], then featureA comes after featureB in tree , i.e. AnB=A!=B and thus A -->B but B -/-> A
        dict_implications = {feature: [] for feature in df_features["feature"]}
        # dict of features that are always asked for the same group of patients and thus can be shuffled : AnB=A=B i.e. A --> B and B-->A
        dict_equivalences = {feature: [] for feature in df_features["feature"]}
        for feature_A in df_features["feature"]:
            A = df_features[df_features["feature"] == feature_A]["patient_set"].item()
            for feature_B in df_features["feature"]:
                B = df_features[df_features["feature"] == feature_B][
                    "patient_set"
                ].item()
                if feature_A != feature_B:
                    if ((A & B) == A) and ((A & B) == B):
                        # A and B are equivalent
                        dict_equivalences[feature_B].append(feature_A)
                    elif (A & B) == A:
                        dict_implications[feature_B].append(feature_A)

        return complete_columns, dict_equivalences, dict_implications

    def get_order_features(self):
        """
        get the order of the features for each patient
        :return: dict with patients as keys and ordered list of features as values
        """
        order_features = {patient: {} for patient in range(len(self))}

        for patient in range(len(self)):
            # get non nan features
            non_nan_features = list(
                self.raw_data.loc[patient][self.raw_data.notna().loc[patient]].index
            )
            # keep only keys that are present for that patient
            index = 0
            while len(non_nan_features) > 0:
                next_feature = self.get_next_feature(non_nan_features)
                order_features[patient][index] = sorted(next_feature)
                for feature in next_feature:
                    non_nan_features.remove(feature)
                index += 1
        all_paths = tuple(
            tuple(tuple(elem) for elem in order_features[index].values())
            for index in range(len(self))
        )
        unique_paths = set(all_paths)
        return order_features

    def get_next_feature(self, non_nan_features):
        """
        Greedily retrieve the order of the features in the non missing features
        :param non_nan_features: non missing features
        :return: features that are not implied by any other features (i.e. root of subtree)
        """
        # get the features implied by each feature in non nan feature
        dict_implications_patient = {
            key: self.dict_implications[key].copy()
            for key in non_nan_features
            if key in self.dict_implications.keys()
        }
        # for each feature in dict_implications_patient.keys() drop the features in its implication list that are missing
        for key in dict_implications_patient.keys():
            for value in dict_implications_patient[key]:
                if value not in non_nan_features:
                    dict_implications_patient[key].remove(value)
        # get the features that are implied by some other feature in the non missing features
        features_implied_by_other = []
        for key in dict_implications_patient.keys():
            features_implied_by_other.extend(dict_implications_patient[key])
        features_implied_by_other = set(features_implied_by_other).intersection(
            non_nan_features
        )
        # get the features that are not in features_implied_by_other, i.e. the features that are the highest in the subtree
        next_feature = list(set(non_nan_features) - features_implied_by_other)

        return next_feature

    def group_patients_by_order_features(self):
        """
        create a nested dictionary with the levels in the tree as keys, and as values a dictionnary with the possible features at that
        level in the tree and a list of patients with values for these features at that level in the tree
        {level_0 : {(feature_1, feature_2) : [patient_1, patient_3] }
        :return: nested dictionary
        """
        # longest path in tree
        max_depth = max(
            [
                len(self.order_features[patient])
                for patient in self.order_features.keys()
            ]
        )
        # find different orders of questions
        df_question_levels = pd.DataFrame()
        # get the possible features in each level in the tree
        for question_level in range(max_depth):
            # find different groups of features
            df_question_levels["level_" + str(question_level)] = [
                self.order_features[patient][question_level]
                if len(self.order_features[patient]) > question_level
                else []
                for patient in self.order_features.keys()
            ]
        # create the nested dict
        question_group_dict = {level: {} for level in range(max_depth)}
        for index, column in enumerate(df_question_levels.columns):
            question_groups = df_question_levels[column].apply(tuple).unique()
            for group in question_groups:
                if group:
                    # for each level in the tree for each feature group in that level, get the list of patients with values
                    question_group_dict[index][group] = [
                        patient
                        for patient in range(len(self))
                        if df_question_levels.loc[patient, column] == list(group)
                    ]

        return question_group_dict

    def remap_test_valid_set(self):
        """map indices in raw df to indices in train, test, valid tensors. Stores mapping and inverse mapping."""
        # mappings with keys corresponding to indices in raw data and values to indices in train/valid/test data respectively
        self.traininig_data_indices_mapping = {
            self.training_data.indices[index]: index
            for index in range(len(self.training_data.indices))
        }
        self.validation_data_indices_mapping = {
            self.validation_data.indices[index]: index
            for index in range(len(self.validation_data.indices))
        }
        self.testing_data_indices_mapping = {
            self.testing_data.indices[index]: index
            for index in range(len(self.testing_data.indices))
        }
        # reverse mappings with keys corresponding to indices in train/valid/test data respectively and values to indices in raw data
        self.traininig_data_indices_reverse_mapping = {
            value: key for key, value in self.traininig_data_indices_mapping.items()
        }
        self.validation_data_indices_reverse_mapping = {
            value: key for key, value in self.validation_data_indices_mapping.items()
        }
        self.testing_data_indices_reverse_mapping = {
            value: key for key, value in self.testing_data_indices_mapping.items()
        }
        group_order_train_features = {
            level: {
                feature_group: []
                for feature_group in self.group_order_features[level].keys()
            }
            for level in self.group_order_features.keys()
        }
        group_order_valid_features = {
            level: {
                feature_group: []
                for feature_group in self.group_order_features[level].keys()
            }
            for level in self.group_order_features.keys()
        }
        group_order_test_features = {
            level: {
                feature_group: []
                for feature_group in self.group_order_features[level].keys()
            }
            for level in self.group_order_features.keys()
        }
        self.order_train_features = {
            self.traininig_data_indices_mapping[patient]: copy.deepcopy(
                self.order_features[patient]
            )
            for patient in self.training_data.indices
        }
        self.order_test_features = {
            self.testing_data_indices_mapping[patient]: copy.deepcopy(
                self.order_features[patient]
            )
            for patient in self.testing_data.indices
        }
        self.order_valid_features = {
            self.validation_data_indices_mapping[patient]: copy.deepcopy(
                self.order_features[patient]
            )
            for patient in self.validation_data.indices
        }
        for level in self.group_order_features.keys():
            for feature_group in self.group_order_features[level].keys():
                for patient in self.group_order_features[level][feature_group]:
                    if patient in self.training_data.indices:
                        group_order_train_features[level][feature_group].append(
                            self.traininig_data_indices_mapping[patient]
                        )
                    elif patient in self.validation_data.indices:
                        group_order_valid_features[level][feature_group].append(
                            self.validation_data_indices_mapping[patient]
                        )
                    elif patient in self.testing_data.indices:
                        group_order_test_features[level][feature_group].append(
                            self.testing_data_indices_mapping[patient]
                        )
        self.train_group_order_features = group_order_train_features
        self.valid_group_order_features = group_order_valid_features
        self.test_group_order_features = group_order_test_features
        # find unique paths in test data
        test_order_features = {
            key: self.order_features[value]
            for key, value in self.testing_data_indices_reverse_mapping.items()
        }
        train_order_features = {
            key: self.order_features[value]
            for key, value in self.traininig_data_indices_reverse_mapping.items()
        }

        all_paths_test = tuple(
            tuple(tuple(elem) for elem in test_order_features[index].values())
            for index in range(len(test_order_features))
        )
        unique_paths_test = set(all_paths_test)
        all_paths_train = tuple(
            tuple(tuple(elem) for elem in train_order_features[index].values())
            for index in range(len(train_order_features))
        )
        unique_paths_train = set(all_paths_train)
        paths_in_test_but_not_in_train = [
            elem for elem in unique_paths_test if elem not in unique_paths_train
        ]
        self.unique_paths_test = {
            patient: all_paths_test[patient]
            for patient in self.testing_data_indices_reverse_mapping.keys()
            if all_paths_test[patient] in paths_in_test_but_not_in_train
        }
        return
