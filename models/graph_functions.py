# functions to produce different plots

import math

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import torch

from numpy.core.function_base import linspace



def plot_heatmap_after_encoder_epoct_all_steps(
    disease_decoder_output, y_test, patients, list_features, qst
):
    """
    Plot values outputed by categorical disease decoders
    :param disease_decoder_output: output of disease decoder
    :param y_test: true values
    :param patients: indices of patients to plot
    :param list_features: list of possible features
    :param qst: questionnaire object
    :return: figure
    """

    disease_mapping = {
        "dxfinal_anemia": "Anemia",
        "dxfinal_dehyd": "Dehydration",
        "dxfinal_diarrhea": "Diarrhea",
        "dxfinal_fws": "FWS",
        "dxfinal_malaria": "Malaria",
        "dxfinal_malnut": "Malnutrition",
        "dxfinal_pna": "Pneumonia",
        "dxfinal_urti": "URTI",
    }

    p = sns.diverging_palette(220, 20, as_cmap=True)
    sns.palplot([p(x) for x in np.linspace(0, 1, 10)])

    def triangle_cdf(x):
        if x < 0.5:
            return -2 * (-x + x**2)
        else:
            return 1 - 2 * x + 2 * x**2

    def p2(x):
        return "rgba" + str(p(triangle_cdf(x)))

    sns.palplot([p2(x) for x in np.linspace(0, 1, 10)])

    diseases = []

    for index, disease in enumerate(qst.disease_names):
        if y_test[patients, index] == 1:
            diseases.append(disease_mapping[disease])
    if len(diseases) == 0:
        diseases.append("No Disease")

    x = list_features[1:]
    x_cleaned = x.copy()
    y = [disease_mapping[name] for name in qst.disease_names]
    z = np.array(disease_decoder_output.T)
    hovertext = list()
    for yi, yy in enumerate(y):
        hovertext.append(list())

    x_axis = ["    " + "No information"]
    x_axis.extend(
        [
            "    "
            + x_cleaned[index]
            + " : "
            + str(
                np.around(
                    qst.raw_data.loc[
                        qst.testing_data_indices_reverse_mapping[patients]
                    ][feature],
                    decimals=2,
                )
            )
            for index, feature in enumerate(x)
        ]
    )
    y_axis = [
        str(elem) if elem not in diseases else "<b>" + "*" + str(elem).upper() + "</b>"
        for elem in y
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_axis,
            y=y_axis,
            coloraxis="coloraxis",
            xgap=0,
            hoverinfo="text",
            text=hovertext,
            zmin=0,
            zmax=1,
            zauto=False,
        )
    )
    fig.update_layout(
        {
            "coloraxis_cmin": 0,
            "coloraxis_cmax": 1,
        }
    )

    fig.update_layout(font_family="Calibri")
    fig.update_layout(coloraxis={"colorscale": [p2(x) for x in linspace(0, 1, 10)]})
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="<b>Probability</b>",
        ),
    )
    fig.update_xaxes(title_text="<b>Sequentially encoded features with values</b>")
    fig.update_yaxes(title_text="<b>Model prediction</b>")
    fig.update_xaxes(tickangle=90, constraintoward="left")

    fig.update_layout(width=1200, height=800)

    fig.write_image("heatmaps/heatmap_epoct_" + str(patients) + ".pdf")
    # fig.show()

    return fig


def plot_calibration(accuracies, confidences):
    """plot calibration curve

    Args:
        accuracies: accuracies of bins
        confidences: confidences of bins
    """
    # matplotlib.rc('xtick', labelsize=20)
    # matplotlib.rc('ytick', labelsize=20)

    sns.set_style("whitegrid")
    pal = sns.color_palette()
    colors = [None, pal[0], pal[2], pal[3]]
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["CMU Serif"],
        }
    )

    plt.figure(figsize=(4, 4))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.7)
    plt.plot(confidences, accuracies, marker="s", color=colors[1], label="MoDN")

    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.legend()
    plt.xticks()
    plt.yticks()
    # plt.grid()

    plt.savefig(f"saved_plots/calibration.pdf", bbox_inches="tight")

    plt.clf()
    return
