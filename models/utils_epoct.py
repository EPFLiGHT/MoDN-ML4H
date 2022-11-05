# utility functions for e-POCT data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def process_features(data):
    """
    process the complete epoct data to select the features and targets to keep
    :param data: complete epoct data
    :return: the data frame of selected features and a dictionnary with the categorical features
    """
    df = data.copy()
    # drop columns with only nan values
    df.dropna(how="all", axis=1, inplace=True)

    # keep columns in dem, lab, signs, symp, dxfinal
    columns_to_keep = []
    # diagnostics
    columns_to_keep.extend(df.filter(regex="^dxfinal").columns)
    # demographics
    columns_to_keep.extend(df.filter(regex="dem_(agem|sex)").columns)
    # history
    columns_to_keep.extend(df.filter(regex="^hist").columns)
    columns_to_keep.extend(df.filter(regex="^lab_\w*_(d0|01)").columns)
    columns_to_keep.extend(df.filter(regex="lab_\w*_cx").columns)
    columns_to_keep.extend(df.filter(regex="^sign\w*_d0").columns)
    columns_to_keep.extend(df.filter(regex="^symp_\w*_d0").columns)
    columns_to_keep.extend(df.filter(regex="sympcc_(?!(01|d3|d7|other))").columns)

    # columns to drop
    columns_to_drop = []
    columns_to_drop.extend(df.filter(regex="date").columns)
    columns_to_drop.extend(df.filter(regex="cat").columns)
    columns_to_drop.extend(df.filter(regex="hb(?!(_d0|ss))").columns)
    columns_to_drop.extend(df.filter(regex="hr(?!(1_d0))\d").columns)
    columns_to_drop.extend(df.filter(regex="rr(?!(1_d0))\d").columns)
    columns_to_drop.extend(df.filter(regex="temp(?!(_d0))").columns)
    columns_to_drop.extend(df.filter(regex="weight(?!(_d0))").columns)
    columns_to_drop.extend(df.filter(regex="lab_other").columns)
    columns_to_drop.extend(["lab_malaria_pcrload_d0"])

    # keep columns
    # drop duplicates
    columns_to_keep = list(set(columns_to_keep))
    df = df[columns_to_keep].copy()

    # drop columns
    for column in columns_to_drop:
        if column in df.columns:
            df.drop(columns=column, inplace=True)

    columns_to_process = [
        "lab_bcx_id_d0",
        "sign_dehyd_skin_d0",
        "hist_pmh",
        "lab_urine_cx_id",
        "symp_complaint_o_d0",
        "dxfinal_hosp",
    ]
    for column in columns_to_process:
        df = column_preprocessing(df, column)
    # drop columns with too low std
    std_dev = df.std()
    # drop features with too low std
    df.drop(columns=std_dev[std_dev < 0.01].index, inplace=True)
    # drop very correlated features

    df = correlations(df, 0.5)

    categorical_features_dict = find_type_of_feature(df)

    # change dem_sex values : 1(male)-->0, 2(female)-->1
    if "dem_sex" in df.columns:
        df["dem_sex"].replace({1: 0, 2: 1}, inplace=True)
    for column in df.columns:
        if df[column].dtype == "object":
            print(column)
            raise ValueError("Feature of type object")
    return df, categorical_features_dict


def convert_strings_to_categories(df, column):
    """
    create mapping between string category names in columns and numeric value to represent that category
    :param df: data frame
    :param column: column in df to process
    :return: dict with the different string categories as keys and an integer corresponding to the numeric value we
    will replace the name with
    """
    mapping_dict = dict(zip(df[column].unique(), range(len(df[column].unique()))))

    if np.nan in mapping_dict.keys():
        mapping_dict[np.nan] = np.nan
    return mapping_dict


def column_preprocessing(df, column):
    """
    convert columns with different string categories as values to numeric categories
    :param df:
    :param column:
    :return:
    """
    if column in df.columns:
        if column in [
            "lab_bcx_id_d0",
            "sign_dehyd_skin_d0",
            "hist_pmh",
            "lab_urine_cx_id",
            "symp_complaint_o_d0",
            "dxfinal_hosp",
        ]:
            mapping_dict = convert_strings_to_categories(df, column)
            df[column] = df[column].replace(mapping_dict)
            df = df.astype({column: "float64"})

    return df


def find_type_of_feature(df):
    """
    find features that are categorical
    :param df: dataframe
    :return: dictionnary with categorical features as keys and the number of different categories as values
    """
    types = df.dtypes
    unique_values = df.nunique()
    categorical_features = {}

    for feature in df.columns:
        if unique_values[feature] < 3 or types[feature] == "int64":
            categorical_features[feature] = unique_values[feature]

    return categorical_features


def get_targets(df, categorical_features_dict):
    """
    separate data frame into features and targets
    :param df: data frame
    :param categorical_features_dict: dictionnary with categorical features as keys
    :return: df of features, df of targets and dictionnary with categorical features
    """
    target_labels = df.filter(regex="dxfinal|out_cure_|out_death_|out_fail_01").columns
    well_processed_targets = []
    very_unbalanced_targets = []
    targets_to_process = []
    num_patients = len(df)
    # find proportion in each target class
    for target in target_labels:
        number_zeros = len(df[target][df[target] == 0])
        number_ones = len(df[target][df[target] == 1])
        if number_zeros + number_ones == num_patients:
            if number_ones / num_patients < 0.05 or number_zeros / num_patients < 0.05:
                very_unbalanced_targets.append(target)
            else:
                well_processed_targets.append(target)
        else:
            targets_to_process.append(target)
    # drop targets with too unbalanced categories
    targets = df[well_processed_targets].copy()
    # drop severe disease
    if "dxfinal_severe" in targets.columns:
        targets.drop(columns=["dxfinal_severe"], inplace=True)
    features = df.drop(columns=target_labels)
    for target in target_labels:
        categorical_features_dict.pop(target, None)

    return features, targets, categorical_features_dict


def correlations(data, threshold):
    """
    compute correlations between each pair of columns of df and drop one of them
    if their correlation is > threshold
    :param data: dataframe
    :param threshold: threshold above which we drop the features if they are too correlated
    :return:
    """
    df = data[sorted(data.columns)].copy()
    dropped_features = []
    print(f"Number of features before dropping highly correlated {df.shape[1]}")
    final_value = threshold
    # compute correlations
    while final_value >= threshold:
        corr_matrix = df.corr().abs()
        triangular = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)
        )
        # find pair of most correlated elements
        max_value = 0.0
        for column in sorted(triangular.columns):
            max_value_col = triangular[column].max()
            if max_value_col > max_value:
                max_value = max_value_col
                max_id = triangular[column].idxmax()
                max_column = column
        if len(df[max_column].dropna()) > len(df[max_id].dropna()):
            df.drop(columns=[max_id], inplace=True)
            dropped_features.append(max_id)
        else:
            df.drop(columns=[max_column], inplace=True)
            dropped_features.append(max_column)
        final_value = max_value
    print(f"Number of features after dropping highly correlated {df.shape[1]}")
    # print(f' Dropped features : {dropped_features}')

    return df
