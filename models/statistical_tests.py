# produce barcharts for model comparison after having performed 5x2 CV
# and centralized training + fine-tuning/compartmentalisation. Uses
# metrics saved in the saved_objects and updtated_centrlized folders

import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import seaborn as sns
import pandas as pd

# metrics with or without feature decoding
FeatureDecoding = False

sns.set_style("whitegrid")

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["CMU Serif"],
    }
)

pal = sns.color_palette()
colors = [None, pal[0], pal[2], pal[3]]


def compute_t(f1_a, f1_b):
    s = 0
    for i in range(5):
        p_1 = f1_a[i, 0] - f1_b[i, 0]
        p_2 = f1_a[i, 1] - f1_b[i, 1]
        p_bar = (p_1 + p_2) / 2
        s += (p_1 - p_bar) ** 2 + (p_2 - p_bar) ** 2
    t = (f1_a[0, 0] - f1_b[0, 0]) / np.sqrt(1 / 5 * s)

    return t


def get_values(results, disease):
    values = np.empty((5, 2))
    for i in range(5):
        for j in range(2):
            values[i, j] = results[i][j][results[i][j]["Disease"] == disease][
                "Macro F_1"
            ].item()
    return values


def CI(array):
    mean = array.mean()
    std = array.std()
    n = len(array)
    # critical value for normal distribution
    c = 1.96
    CI = c * std / np.sqrt(n)
    return CI


# f1 scores results
if FeatureDecoding:
    metrics = pd.read_pickle("models/saved_objects/metrics_decoding.p")
else:
    metrics = pd.read_pickle("models/saved_objects/metrics.p")

macro_f1_mlp = metrics["mlp"]["f1"]
results_mlp = metrics["mlp"]["results"]
macro_f1_log = metrics["log_reg"]["f1"]
results_log = metrics["log_reg"]["results"]
macro_f1_modn = metrics["modn"]["f1"]
results_modn = metrics["modn"]["results"]
t_log = compute_t(macro_f1_modn, macro_f1_log)
p_log = np.round(scipy.stats.t.sf(abs(t_log), df=5), 5)
t_mlp = compute_t(macro_f1_modn, macro_f1_mlp)
p_mlp = np.round(scipy.stats.t.sf(abs(t_mlp), df=5), 5)
print(f"t-score logistic regression versus modn {t_log}, pvalue {p_log}")
print(f"t-score mlp versus modn {t_mlp}, pvalue {p_mlp}")
fig, ax = plt.subplots(figsize=(4, 4))
p_values_log = []

for disease in results_modn[0][0]["Disease"]:
    array_1 = get_values(results_modn, disease)
    array_2 = get_values(results_log, disease)
    t = compute_t(array_1, array_2)
    p = np.round(scipy.stats.t.sf(abs(t), df=5), 5)
    print(f"{disease} : t : {t}, p : {p}")
    p_values_log.append(p)

p_values_mlp = []
for disease in results_modn[0][0]["Disease"]:
    array_1 = get_values(results_modn, disease)
    array_2 = get_values(results_mlp, disease)
    t = compute_t(array_1, array_2)
    p = np.round(scipy.stats.t.sf(abs(t), df=5), 5)
    print(f"{disease} : t : {t}, p : {p}")
    p_values_mlp.append(p)

print(
    f"modn mean {macro_f1_modn.mean()}, log reg mean {macro_f1_log.mean()}, mlp mean {macro_f1_mlp.mean()}"
)

barWidth = 0.35
barWidthFirst = 0.35
# Set position of bar on X axis
r1_first = 0
r2_first = 0 + barWidthFirst
r3_first = r2_first + barWidthFirst
r1 = np.arange(1, len(results_modn[0][0]["Disease"]) + 1) * -1.35
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]


def add_stats(value_1, value_2, y_place_1, y_place_2, p, h=0.1, gap_1=0.01, gap_2=0.01):
    text_gap = 0.005
    # define text
    if p >= 0.05:
        text = "ns"
        text_gap = 0.015
    elif p >= 0.005:
        text = "*"
    elif p >= 0.001:
        text = "**"
    else:
        text = "***"
    plt.plot(
        [value_1 + gap_1, value_1 + h, value_1 + h, value_2 + gap_2],
        [y_place_1, y_place_1, y_place_2, y_place_2],
        lw=0.5,
        alpha=0.6,
        c="black",
    )
    plt.text(
        value_1 + h + text_gap,
        (y_place_1 + y_place_2) * 0.5,
        text,
        ha="center",
        va="center",
        alpha=0.6,
        color="black",
        fontsize=8,
        rotation=-90,
    )
    return


ax.barh(r1_first, macro_f1_modn.mean(), color=colors[1], height=barWidthFirst)
ax.barh(r2_first, macro_f1_mlp.mean(), color=colors[2], height=barWidthFirst)
ax.barh(r3_first, macro_f1_log.mean(), color=colors[3], height=barWidthFirst)

add_stats(
    macro_f1_modn.mean(), macro_f1_mlp.mean(), r1_first, r2_first, p=p_mlp, h=0.05
)
add_stats(
    macro_f1_modn.mean(),
    macro_f1_log.mean(),
    r1_first,
    r3_first,
    p=p_log,
    h=0.1,
    gap_1=0.08,
)

for index, disease in enumerate(results_modn[0][0]["Disease"]):
    values_mlp = get_values(results_mlp, disease)
    values_log = get_values(results_log, disease)
    values_modn = get_values(results_modn, disease)
    # values_knn = get_values(results_knn, disease)
    if index == 0:
        ax.barh(
            r1[index],
            width=values_modn.mean(),
            color=colors[1],
            height=barWidth,
            label="MoDN (Ours)",
        )
        ax.barh(
            r2[index],
            width=values_mlp.mean(),
            color=colors[2],
            height=barWidth,
            label="Multilayer Perceptron",
        )
        ax.barh(
            r3[index],
            width=values_log.mean(),
            color=colors[3],
            height=barWidth,
            label="Logistic regression",
        )

        add_stats(
            values_modn.mean(),
            values_mlp.mean(),
            r1[index],
            r2[index],
            p=p_values_mlp[index],
            h=0.05,
        )
        add_stats(
            values_modn.mean(),
            values_log.mean(),
            r1[index],
            r3[index],
            p=p_values_log[index],
            h=0.1,
            gap_1=0.08,
        )

    else:
        ax.barh(
            r1[index],
            width=values_modn.mean(),
            color=colors[1],
            height=barWidth,
        )
        ax.barh(
            r2[index],
            width=values_mlp.mean(),
            color=colors[2],
            height=barWidth,
        )
        ax.barh(
            r3[index],
            width=values_log.mean(),
            color=colors[3],
            height=barWidth,
        )

        add_stats(
            values_modn.mean(),
            values_mlp.mean(),
            r1[index],
            r2[index],
            p=p_values_mlp[index],
            h=0.05,
        )
        add_stats(
            values_modn.mean(),
            values_log.mean(),
            r1[index],
            r3[index],
            p=p_values_log[index],
            h=0.1,
            gap_1=0.08,
        )

y_ticks = ["\\textbf{All diseases}\n(aggregated)"]
y_ticks.extend(
    list(
        results_modn[0][0]["Disease"].replace(
            {
                "dxfinal_anemia": "Anemia",
                "dxfinal_dehyd": "Dehydration",
                "dxfinal_diarrhea": "Diarrhea",
                "dxfinal_fws": "FWS",
                "dxfinal_malaria": "Malaria",
                "dxfinal_malnut": "Malnutrition",
                "dxfinal_pna": "Pneumonia",
                "dxfinal_urti": "URTI",
            }
        )
    )
)

# *: p<0.05, **: p<0.005, ***: p<0.001, ns: not significant
textstr = "\n".join(
    (
        "\\makebox[2em][l]{*} $p<0.05$",
        "\\makebox[2em][l]{**} $p < 0.005$",
        "\\makebox[2em][l]{***} $p < 0.001$",
        "\\makebox[2em][l]{ns} not significant",
    )
)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle="round", facecolor="white", alpha=0.13, edgecolor="black")

# place a text box in upper left in axes coords
ax.text(
    1.035,
    0.73,
    textstr,
    transform=ax.transAxes,
    # fontsize=14,
    verticalalignment="top",
    bbox=props,
)
y_ticks_placement = [r1_first + barWidthFirst]
y_ticks_placement.extend([r + barWidth for r in r1])

plt.yticks(y_ticks_placement, y_ticks, ha="left")
plt.tick_params(axis="y", pad=70)
plt.xlabel("Macro F1 score")
ax.tick_params(axis="both")
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(1, 1))
sns.despine(bottom=True, top=True, right=True)
plt.grid(False, axis="y")
if FeatureDecoding:
    plt.savefig("models/saved_plots/f1_scores_decoding.pdf", bbox_inches="tight")
else:
    plt.savefig("models/saved_plots/f1_scores.pdf", bbox_inches="tight")


# plots for distributed training new

local_training = np.load("models/updated_centralized/lower-baseline.npy")
central_training_100 = np.load("models/updated_centralized/common_100.npy")
fine_tune_100 = np.load("models/updated_centralized/fine-tune_100.npy")
central_training_90 = np.load("models/updated_centralized/common_90.npy")
fine_tune_90 = np.load("models/updated_centralized/fine-tune_90.npy")
central_training_80 = np.load("models/updated_centralized/common_80.npy")
fine_tune_80 = np.load("models/updated_centralized/fine-tune_80.npy")
central_training_60 = np.load("models/updated_centralized/common_60.npy")
fine_tune_60 = np.load("models/updated_centralized/fine-tune_60.npy")
upper_baseline_60 = np.load("models/updated_centralized/upper-baseline_60.npy")
upper_baseline_80 = np.load("models/updated_centralized/upper-baseline_80.npy")
upper_baseline_90 = np.load("models/updated_centralized/upper-baseline_90.npy")
upper_baseline_100 = np.load("models/updated_centralized/upper-baseline_100.npy")
compartm_60 = np.load("models/updated_centralized/compartm_60.npy")
compartm_80 = np.load("models/updated_centralized/compartm_80.npy")
compartm_90 = np.load("models/updated_centralized/compartm_90.npy")
compartm_100 = np.load("models/updated_centralized/common_100.npy")

# combined plot

central_training_means = [
    central_training_60.mean(),
    central_training_80.mean(),
    central_training_100.mean(),
]
print(f"Central training means {central_training_means}")
central_training_stds = [
    central_training_60.std(),
    central_training_80.std(),
    central_training_100.std(),
]
central_training_CI = [
    CI(central_training_60),
    CI(central_training_80),
    CI(central_training_100),
]
fine_tune_means = [fine_tune_60.mean(), fine_tune_80.mean(), fine_tune_100.mean()]
print(f"Fine-tuning means {fine_tune_means}")
fine_tune_stds = [fine_tune_60.std(), fine_tune_80.std(), fine_tune_100.std()]
fine_tune_CI = [CI(fine_tune_60), CI(fine_tune_80), CI(fine_tune_100)]
compartm_means = [compartm_60.mean(), compartm_80.mean(), compartm_100.mean()]
compartm_stds = [compartm_60.std(), compartm_80.std(), compartm_100.std()]
compartm_CI = [CI(compartm_60), CI(compartm_80), CI(compartm_100)]
upper_baseline_means = [
    upper_baseline_60.mean(),
    upper_baseline_80.mean(),
    upper_baseline_100.mean(),
]
upper_baseline_stds = [
    upper_baseline_60.std(),
    upper_baseline_80.std(),
    upper_baseline_100.std(),
]
upper_baseline_CI = [
    CI(upper_baseline_60),
    CI(upper_baseline_80),
    CI(upper_baseline_100),
]

fig, ax = plt.subplots(figsize=(6, 4))
# matplotlib.rcParams.update({"font.size": 9})

ind = np.array([0.25, 1, 2, 3])  # the x locations for the groups
width = 0.15  # the width of the bars
ax.bar(
    ind[0],
    local_training.mean(),
    width,
    yerr=CI(local_training),
    label="Local",
    color="#71D78B",
    capsize=2,
)
ax.bar(
    ind[1:],
    central_training_means,
    width,
    yerr=central_training_CI,
    label="Static",
    color="#71D4D7",
    capsize=2,
)
ax.bar(
    ind[1:] + width,
    fine_tune_means,
    width,
    yerr=fine_tune_CI,
    label="Modularised fine-tune",
    color="#616A6B",
    capsize=2,
)
ax.bar(
    ind[1:] + 2 * width,
    compartm_means,
    width,
    yerr=compartm_CI,
    label="Modularised update",
    color="#AEB6BF",
    capsize=2,
)
ax.bar(
    ind[1:] + 3 * width,
    upper_baseline_means,
    width,
    yerr=upper_baseline_CI,
    label="Global",
    color="#B384DA",
    capsize=2,
)
ax.axhline(y=local_training.mean(), color="darkslategrey", linestyle="--", linewidth=1)
# ax.set_title('Decentralized pre-training')
x_ticks = [elem + width * 1.5 if index > 0 else elem for index, elem in enumerate(ind)]
ax.set_xticks(
    x_ticks,
    labels=[
        "Local \n training",
        "60\% of shared\n features",
        "80\% of shared\n features",
        "100\% of shared\n features",
    ],
)
ax.set_ylabel("Macro F1 score")
ax.legend(loc="upper left")
ax.grid(False, axis="x")
sns.despine(left=True, right=True, top=True)
plt.ylim((0.5, 0.8))
ax.autoscale_view()
plt.savefig("models/saved_plots/combined_barchart.pdf", bbox_inches="tight")
