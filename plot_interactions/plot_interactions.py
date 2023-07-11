#!/usr/bin/env python

"""
.. module: plot_interactions
   :platform: Linux, Windows
   :synopsis: Script for plotting interactions between cell types

.. moduleauthor:: Fredrik Nysjo
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

import math
import sys
import os


#INPUT_FILENAMES = [
#    "results_TMA5_Panel1.csv",
#    "results_TMA6_Panel1.csv",
#    "results_TMA7_Panel1.csv",
#    "results_TMA8_Panel1.csv"
#]
#CLASSES = ["Astrocyte", "Glioma", "Neuron", "TAMM", "Endothelial", "Negative"]
INPUT_FILENAMES = [
    "results_TMA5_Panel2.csv",
    "results_TMA6_Panel2.csv",
    "results_TMA7_Panel2.csv",
    "results_TMA8_Panel2.csv"
]
CLASSES = ["Astrocyte", "Glioma", "Oligodendrocyte", "Leukocyte", "TAMM", "Endothelial", "PVC", "Negative"]
TMA_ROWS = range(1, 13, 2)
TMA_COLS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
OUTPUT_DIR = "output"
SHOW_PAIR_LABELS = True
SHOW_FIGURES = False
INCLUDE_ENRICHMENT_SCORES = True
INCLUDE_ABUNDANCE_FRACTION = True


def main():
    measurements = []
    for i in range(0, len(CLASSES)):
        if INCLUDE_ENRICHMENT_SCORES:
            for j in range(0, len(CLASSES)):
                measurements.append(CLASSES[i] + "_" + CLASSES[j] + "_30")
                measurements.append(CLASSES[i] + "_" + CLASSES[j] + "_50")
        if INCLUDE_ABUNDANCE_FRACTION:
            measurements.append(CLASSES[i] + "_abundance")
            measurements.append(CLASSES[i] + "_fraction")

    xs_byIndex = [[] for i in range(0, len(measurements))]
    ys_byIndex = [[] for i in range(0, len(measurements))]

    panel_name = INPUT_FILENAMES[0].split("_")[2].strip(".csv")
    for input_filename in INPUT_FILENAMES:
        # Read CSV containing per-core measurements for all cores in the TMA
        data = pd.read_csv(input_filename, sep=",")
        print(data.shape)

        # Extract image names for all of the cores
        cores = list(data["Name"])

        # Sort out valid cases (i.e., pairs of observations) in the data. Assume that
        # the cores have been stored in sorted order (by image name) in the CSV!
        cases = []
        labels = []
        for y in TMA_ROWS:
            for x in TMA_COLS:
                s0 = "," + str(y + 0) + "," + x  # Substring to identify first observation
                s1 = "," + str(y + 1) + "," + x  # Substring to identify second observation
                case = []
                for index, name in enumerate(cores):
                    if name.count(s0) or name.count(s1):
                        case.append((index, name))
                if len(case) == 2:
                    cases.append(case)
                    labels.append(s0.split(",")[1] + "-" + s1[1:])

        # Extract measurements for the valid cases
        for index, colName in enumerate(measurements):
            values = data[colName]
            xs = [values[case[0][0]] for case in cases]
            ys = [values[case[1][0]] for case in cases]
            xs_byIndex[index] += xs
            ys_byIndex[index] += ys

        # Plot valid cases in per-measurement scatter plots
        for index, colName in enumerate(measurements):
            fig = plt.figure(index)
            values = data[colName]
            xs = [values[case[0][0]] for case in cases]
            ys = [values[case[1][0]] for case in cases]
            plt.scatter(xs, ys, label=input_filename.strip(".csv").strip("results_"))
            plt.axis("square")
            plt.ylabel("Observation 1")
            plt.xlabel("Observation 2")
            plt.legend(loc="upper right")
            plt.title(colName)
            plt.axhline(linewidth="1.0", linestyle="solid", color="#aaa", zorder=-99)
            plt.axvline(linewidth="1.0", linestyle="solid", color="#aaa", zorder=-99)
            if SHOW_PAIR_LABELS:
                for j, s in enumerate(labels):
                    fig.gca().annotate(s, (xs[j], ys[j]), fontsize="small", alpha=0.5)

    # Compute statistics about the measurement data
    stats = {"corrcoef": {}, "median": {}, "mean": {}, "std": {}}
    for index, colName in enumerate(measurements):
        corr = np.ma.corrcoef(np.ma.masked_invalid(xs_byIndex[index]),
                              np.ma.masked_invalid(ys_byIndex[index]))
        stats["corrcoef"][colName] = [corr[0, 1]]
        stats["median"][colName] = [np.ma.median(np.ma.masked_invalid(
            [(x + y) * 0.5 for x, y in zip(xs_byIndex[index], ys_byIndex[index])]))]
        stats["mean"][colName] = [np.ma.mean(np.ma.masked_invalid(
            [(x + y) * 0.5 for x, y in zip(xs_byIndex[index], ys_byIndex[index])]))]
        stats["std"][colName] = [np.ma.std(np.ma.masked_invalid(
            [(x + y) * 0.5 for x, y in zip(xs_byIndex[index], ys_byIndex[index])]))]

    # Generate directory for outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Add correlation coefficients to and save figures
    for index, colName in enumerate(measurements):
        corr = np.ma.corrcoef(np.ma.masked_invalid(xs_byIndex[index]),
                              np.ma.masked_invalid(ys_byIndex[index]))
        plt.figure(index)
        plt.title(colName + ", Corr=" + str(stats["corrcoef"][colName][0]))
        plt.savefig(os.path.join(OUTPUT_DIR, colName + ".png"))

    # Save other output data
    pd.DataFrame.from_dict(stats["corrcoef"]).to_csv(os.path.join(OUTPUT_DIR, "stats_" + panel_name + "_corrcoef.csv"), sep=",")
    pd.DataFrame.from_dict(stats["median"]).to_csv(os.path.join(OUTPUT_DIR, "stats_" + panel_name + "_median.csv"), sep=",")
    pd.DataFrame.from_dict(stats["mean"]).to_csv(os.path.join(OUTPUT_DIR, "stats_" + panel_name + "_mean.csv"), sep=",")
    pd.DataFrame.from_dict(stats["std"]).to_csv(os.path.join(OUTPUT_DIR, "stats_" + panel_name + "_std.csv"), sep=",")

    if SHOW_FIGURES:
        plt.show()


if __name__ == "__main__":
    main()
