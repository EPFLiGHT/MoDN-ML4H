# MLP and logistic regression baseline functions

import random

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from modules import BaselineBinary2layersMLP
from utils import compute_metrics, latex_table_from_df
import wandb

from training_procedures_epoct import training_baseline_mlp

random.seed(0)


def Logistic_regression_EPOCT(
    X_train, y_train, X_test, y_test, penalty=1, targets=None
):
    """
    fits logistic regression with L_2 penalty to epoct data and computes metrics on test set
    :param X_train: training data set
    :param y_train: training targets
    :param X_test: testing data set
    :param y_test: test targets
    :param penalty: L_2 penalty
    :param targets: pandas data frame with targets as values and target names as column names
    :return: macro f_1, results dataframe and class probabilities
    """
    print(f" Baseline assessment with logistic regression : ")
    if len(y_train.shape) == 1:
        clf = LogisticRegression(random_state=0, n_jobs=-1, multi_class="multinomial")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(
            f"\n Logistic regression results on unseen test set : \n {classification_report(y_test, y_pred)}"
        )
    else:
        results = pd.DataFrame(
            columns=["Disease", "Class", "Precision", "Recall", "F_1", "Num_samples"]
        )
        aggregated_results = pd.DataFrame(
            columns=["Disease", "Macro Precision", "Macro Recall", "Macro F_1"]
        )
        f1_macro_all = 0
        probabilities = np.empty(shape=(y_test.shape))
        for index in range(y_train.shape[1]):
            clf = LogisticRegression(
                random_state=0, n_jobs=-1, C=penalty, max_iter=1000
            )
            clf.fit(X_train, y_train[:, index])
            y_pred = clf.predict(X_test)
            probabilities[:, index] = clf.predict_proba(X_test)[:, 1]
            (
                precision_0,
                recall_0,
                F1_0,
                precision_1,
                recall_1,
                F1_1,
                F1_average,
            ) = compute_metrics(y_pred, y_test[:, index])
            results = results.append(
                pd.Series(
                    {
                        "Disease": targets.columns[index],
                        "Class": 0,
                        "Precision": precision_0,
                        "Recall": recall_0,
                        "F_1": F1_0,
                        "Num_samples": len(
                            [
                                elem
                                for elem in range(len(y_test))
                                if y_test[elem, index] == 0
                            ]
                        ),
                    }
                ),
                ignore_index=True,
            )
            results = results.append(
                pd.Series(
                    {
                        "Disease": targets.columns[index],
                        "Class": 1,
                        "Precision": precision_1,
                        "Recall": recall_1,
                        "F_1": F1_1,
                        "Num_samples": len(
                            [
                                elem
                                for elem in range(len(y_test))
                                if y_test[elem, index] == 1
                            ]
                        ),
                    }
                ),
                ignore_index=True,
            )
            aggregated_results = aggregated_results.append(
                pd.Series(
                    {
                        "Disease": targets.columns[index],
                        "Macro Precision": np.around(
                            (precision_0 + precision_1) / 2, decimals=2
                        ),
                        "Macro Recall": np.around(
                            (recall_0 + recall_1) / 2, decimals=2
                        ),
                        "Macro F_1": np.around((F1_1 + F1_0) / 2, decimals=2),
                    }
                ),
                ignore_index=True,
            )
        print(f"F1 macro all : {f1_macro_all}")
        print(results)
        latex_table_from_df(results)
        print(aggregated_results)
        latex_table_from_df(aggregated_results)
        f1 = pd.Series(
            {
                "Macro_F1": results["F_1"].sum() / len(results),
                "Weighted_F1": (results["F_1"] * results["Num_samples"]).sum()
                / results["Num_samples"].sum(),
            }
        )
        print(f1)
        print(f1.to_latex())
    return results["F_1"].sum() / len(results), aggregated_results, probabilities


def MLP(X_train, y_train, X_valid, y_valid, X_test, y_test, targets=None):
    """
    trains and fits a multilayer perceptron to data and compute metrics on test set
    :param X_train: training data set
    :param y_train: training targets
    :param X_valid: validation set
    :param y_valid: validation targets
    :param X_test: testing data set
    :param y_test: test targets
    :param targets: pandas data frame with targets as values and target names as column names
    :return: macro f_1, results dataframe and class probabilities
    """
    print(f" Baseline assessment with MLP : ")

    results = pd.DataFrame(
        columns=["Disease", "Class", "Precision", "Recall", "F_1", "Num_samples"]
    )
    aggregated_results = pd.DataFrame(
        columns=["Disease", "Macro Precision", "Macro Recall", "Macro F_1"]
    )
    f1_macro_all = 0
    training_params = {
        0: {"lr": 0.002, "n_epochs": 200, "batch_size": len(X_train)},
        1: {"lr": 0.002, "n_epochs": 200, "batch_size": len(X_train)},
        2: {"lr": 0.002, "n_epochs": 200, "batch_size": len(X_train)},
        3: {"lr": 0.002, "n_epochs": 300, "batch_size": len(X_train)},
        4: {"lr": 0.002, "n_epochs": 200, "batch_size": len(X_train)},
        5: {"lr": 0.002, "n_epochs": 300, "batch_size": len(X_train)},
        6: {"lr": 0.002, "n_epochs": 200, "batch_size": len(X_train)},
        7: {"lr": 0.002, "n_epochs": 300, "batch_size": len(X_train)},
    }

    probabilities = np.empty(shape=(y_test.shape))
    for index in range(y_train.shape[1]):
        # model = BaselineBinaryMLP(input_size=X_train.shape[1])
        model = BaselineBinary2layersMLP(input_size=X_train.shape[1])
        training_parameters = training_params[index]
        model = training_baseline_mlp(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train[:, index]),
            torch.tensor(X_valid, dtype=torch.float32),
            torch.tensor(y_valid[:, index]),
            model,
            training_parameters,
        )

        output = model(torch.tensor(X_test, dtype=torch.float32))
        probabilities[:, index] = torch.exp(output[:, 1]).detach()
        _, y_pred = torch.max(output, 1)
        (
            precision_0,
            recall_0,
            F1_0,
            precision_1,
            recall_1,
            F1_1,
            F1_average,
        ) = compute_metrics(y_pred, y_test[:, index])
        results = results.append(
            pd.Series(
                {
                    "Disease": targets.columns[index],
                    "Class": 0,
                    "Precision": precision_0,
                    "Recall": recall_0,
                    "F_1": F1_0,
                    "Num_samples": len(
                        [
                            elem
                            for elem in range(len(y_test))
                            if y_test[elem, index] == 0
                        ]
                    ),
                }
            ),
            ignore_index=True,
        )
        results = results.append(
            pd.Series(
                {
                    "Disease": targets.columns[index],
                    "Class": 1,
                    "Precision": precision_1,
                    "Recall": recall_1,
                    "F_1": F1_1,
                    "Num_samples": len(
                        [
                            elem
                            for elem in range(len(y_test))
                            if y_test[elem, index] == 1
                        ]
                    ),
                }
            ),
            ignore_index=True,
        )
        aggregated_results = aggregated_results.append(
            pd.Series(
                {
                    "Disease": targets.columns[index],
                    "Macro Precision": np.around(
                        (precision_0 + precision_1) / 2, decimals=2
                    ),
                    "Macro Recall": np.around((recall_0 + recall_1) / 2, decimals=2),
                    "Macro F_1": np.around((F1_1 + F1_0) / 2, decimals=2),
                }
            ),
            ignore_index=True,
        )
    print(f"F1 macro all : {f1_macro_all}")
    print(results)
    latex_table_from_df(results)
    print(aggregated_results)
    latex_table_from_df(aggregated_results)
    f1 = pd.Series(
        {
            "Macro_F1": results["F_1"].sum() / len(results),
            "Weighted_F1": (results["F_1"] * results["Num_samples"]).sum()
            / results["Num_samples"].sum(),
        }
    )
    print(f1)
    wandb.log({"result_table": aggregated_results, "f1": pd.DataFrame(f1)})
    print(f1.to_latex())
    return results["F_1"].sum() / len(results), aggregated_results, probabilities
