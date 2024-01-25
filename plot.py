import numpy as np
import matplotlib.pyplot as plt
from os.path import isfile
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--no_latex', action='store_true', default=False)
args = parser.parse_args()

# formatting template from https://github.com/pierreablin/python-sessions/blob/master/Pretty_plots.ipynb
fontsize = 18
params = {
    "axes.labelsize": fontsize + 2,
    "font.size": fontsize + 2,
    "legend.fontsize": fontsize + 2,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "text.usetex": not args.no_latex,
}
plt.rcParams.update(params)
plt.rc("font", family="Times New Roman")


X_LIM = {"energy": 650, "pollution": 125}
COST_UB = {"energy": 16, "pollution": 12}
BETA = {"energy": 0.3, "pollution": 0.1}

labels = {
    "energy": ["$m=2$", "$m=4$", "$m=8$", "$\\alpha_e$"],
    "pollution": ["$m=2$", "$m=4$", "$m=8$", "$\\alpha_C$"],
}

for env_type in ["energy", "pollution"]:
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    for i, m in enumerate([2, 4, 8]):
        run_potentials, run_c_values = [], []
        for seed in range(10):
            dir_name = "experiments/" + env_type + "/" + str(m) + "/" + str(seed)
            if isfile(dir_name + "/potential_hist.npy") and isfile(
                dir_name + "/value_c_hist.npy"
            ):
                run_potentials.append(
                    np.load(dir_name + "/potential_hist.npy")[: X_LIM[env_type]]
                )
                run_c_values.append(
                    np.load(dir_name + "/value_c_hist.npy")[: X_LIM[env_type]]
                )
            else:
                continue

        pm = np.mean(np.stack(run_potentials), axis=0)
        scaling = 1 / (pm.max() - pm.min())

        potential_std = np.std(scaling * np.stack(run_potentials), axis=0)
        potential_mean = scaling * (pm - pm.min())

        c_value_mean = np.mean(np.stack(run_c_values), axis=0)[: X_LIM[env_type]]
        c_value_std = np.std(np.stack(run_c_values), axis=0)[: X_LIM[env_type]]

        if env_type == "energy":
            axs[0].set_xscale("log")
            axs[1].set_xscale("log")
        axs[0].plot(potential_mean, label=labels[env_type][i])
        axs[0].fill_between(
            np.arange(0, len(potential_mean)),
            potential_mean - potential_std,
            potential_mean + potential_std,
            alpha=0.3,
        )
        axs[1].plot(c_value_mean, label=labels[env_type][i])
        axs[1].fill_between(
            np.arange(0, len(c_value_mean)),
            c_value_mean - c_value_std,
            c_value_mean + c_value_std,
            alpha=0.3,
        )

    a = axs[0].plot([0] * len(c_value_mean), label=labels[env_type][3])[0]
    axs[1].plot(
        [COST_UB[env_type] + BETA[env_type]] * len(c_value_mean),
        label=labels[env_type][3],
    )

    # adjust y ticks for pollution environment
    if env_type == "pollution":
        start, end = axs[1].get_ylim()
        axs[1].yaxis.set_ticks(np.arange(np.ceil(start), np.ceil(end), 1))

    axs[0].set_xlabel("iteration $t$")
    axs[0].set_ylabel("$\Phi(\pi^{(t)})$")
    axs[1].set_xlabel("iteration $t$")
    axs[1].set_ylabel("$V_c(\pi^{(t)})$")
    lgd = fig.legend(labels=labels[env_type], loc="lower center", ncol=4)
    a.set_visible(False)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.26)

    if not os.path.exists("plots"):
        os.makedirs("plots")
    fig.savefig("plots/" + env_type + ".pdf")
