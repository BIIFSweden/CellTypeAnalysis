#!/usr/bin/env python

"""
.. module:: plot_confusion
   :platform: Linux
   :synopsis: Code for generating confusion matrices for cell type data
.. moduleauthor:: Fredrik Nysjo
"""

import utils

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import sklearn.metrics

import sys
import os


# You can change these settings to modify the behaviour of the script
settings = {
    "dataset": "train",  # Should be "train" or "test"
    "selected_core": -1,  # If value < 0, include all cores in the matrix
    "encoding_predicted": "utf8",  # Should be "utf8" or "iso-8859-15"
    "encoding_annotated": "iso-8859-15",  # Should be "utf8" or "iso-8859-15"
    "xcol": "Centroid X µm",  # CSV column to use for X-coordinates
    "ycol": "Centroid Y µm",  # CSV column to use for Y-coordinates
    "suffix_predicted": "_resolved",  # Suffix used in CSV filenames for predictions
    "suffix_annotated": "_GT",  # Suffix used in CSV filenames for annotatations
}


# You can change this info to adapt the script for other cell types
celltype_info = {
    "mapping_predicted": {  # Mapping from label names in CSV data
        "Astrocyte": "Astrocyte",
        "Glioma": "Glioma",
        "Oligodendrocyte": "Oligodendrocyte",
        "Leukocyte": "Leukocyte",
        "TAMM": "TAMM",
        "Endothelial": "Endothelial",
        "PVC": "PVC",
        "Negative": "Negative",
    },
    "mapping_annotated": {  # Mapping from label names in CSV data
        "Astrocyte": "Astrocyte",
        "Glioma cell": "Glioma",
        "Oligodendrocyte": "Oligodendrocyte",
        "Leukocyte": "Leukocyte",
        "TAMM": "TAMM",
        "Endothelial cell": "Endothelial",
        "PVC": "PVC",
    },
    "labels": {  # Mapping from label names to label index
        "Ambiguous": 0,
        "Astrocyte": 1,
        "Glioma": 2,
        "Oligodendrocyte": 3,
        "Leukocyte": 4,
        "TAMM": 5,
        "Endothelial": 6,
        "PVC": 7,
        "Negative": 8,
    },
    "cellts": [  # Name of cell types, including the ambiguous class
        "Astrocyte",
        "Glioma",
        "Oligodendrocyte",
        "Leukocyte",
        "TAMM",
        "Endothelial",
        "PVC",
        "Negative",
        "Ambiguous",
    ],
}


class Grid:
    def __init__(self, bounds, size=(16, 16)):
        assert len(bounds) == 4
        assert len(size) == 2

        self.size = size
        self.bounds = bounds
        self.points = [[[]] * size[0] for y in range(0, size[1])]
        self.indices = [[[]] * size[0] for y in range(0, size[1])]


def grid_add_points_from_data(grid, data, xlabel, ylabel):
    for i in range(0, len(data)):
        x = data.iloc[i][xlabel]
        y = data.iloc[i][ylabel]
        gx = int((x - grid.bounds[0]) // (grid.bounds[2] - grid.bounds[0] + 1e-6))
        gy = int((y - grid.bounds[1]) // (grid.bounds[3] - grid.bounds[1] + 1e-6))
        grid.points[gy][gx].append((x, y))
        grid.indices[gy][gx].append(i)


def grid_find_closest(grid, p, threshold=0.5):
    closest_index = -1
    closest_dist = 99999.0
    gx = int((p[0] - grid.bounds[0]) // (grid.bounds[2] - grid.bounds[0] + 1e-6))
    gy = int((p[1] - grid.bounds[1]) // (grid.bounds[3] - grid.bounds[1] + 1e-6))
    gx = max(0, min(grid.size[0], gx))
    gy = max(0, min(grid.size[1], gy))
    for q, i in zip(grid.points[gy][gx], grid.indices[gy][gx]):
        dist = ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5
        if dist < closest_dist and dist < threshold:
            closest_dist = dist
            closest_index = i
    return closest_index


def generate_final_class(data, mapping, labels):
    result = []
    for i in range(0, len(data)):
        annotation = data["Name"][i]
        final_class = []
        for key, value in mapping.items():
            if (key in annotation) and not ("Not-" + key in annotation):
                final_class.append(value)
        if len(final_class) > 1:
            final_class = ["Ambiguous"]
        if len(final_class) == 0:
            final_class = ["Negative"]
        result.append(labels[final_class[0]])
    data["Class"] = result


def plot_confusion_matrix(actual, pred, cellts, output_prefix=""):
    assert len(actual) == len(pred)

    cm = utils.myOwnCM(actual, pred, numclasses=len(cellts))
    utils.plot_confusion_matrix(
        cm, cellts, cmap="viridis", save=(output_prefix + "cm.png")
    )


def main():
    # Note: these paths and filenames are currently hardcoded for specific
    # datasets. So you need to change them for new data. Should perhaps be
    # handled in some better way, i.e., by taking folder names as input from
    # the command line (TODO)
    if settings["dataset"] == "train":
        input_prefix = "classified_training/"
        input_prefix2 = "gt_training/"
        output_prefix = "output_training/"
        input_filenames = [
            "5_10B_Panel2_resolved.csv",
            "5_10F_Panel2_resolved.csv",
            "5_10I_Panel2_resolved.csv",
            "5_11I_Panel2_resolved.csv",
            "5_12H_Panel2_resolved.csv",
            "5_1A_Panel2_resolved.csv",
            "5_3C_Panel2_resolved.csv",
            "5_4D_Panel2_resolved.csv",
            "5_9D_Panel2_resolved.csv",
            "7_1E_Panel2_resolved.csv",
        ]
        input_filenames2 = [
            "5_10B_Panel2_GT.csv",
            "5_10F_Panel2_GT.csv",
            "5_10I_Panel2_GT.csv",
            "5_11I_Panel2_GT.csv",
            "5_12H_Panel2_GT.csv",
            "5_1A_Panel2_GT.csv",
            "5_3C_Panel2_GT.csv",
            "5_4D_Panel2_GT.csv",
            "5_9D_Panel2_GT.csv",
            "7_1E_Panel2_GT.csv",
        ]
    elif settings["dataset"] == "test":
        input_prefix = "classified_test/"
        input_prefix2 = "gt_test/"
        output_prefix = "output_test/"
        input_filenames = [
            "5_12E_Panel2_resolved.csv",
            "7_3A_Panel2_resolved.csv",
            "7_5D_Panel2_resolved.csv",
            "7_7B_Panel2_resolved.csv",
            "8_12B_Panel2_resolved.csv",
            "8_1B_Panel2_resolved.csv",
            "8_1C_Panel2_resolved.csv",
            "8_2A_Panel2_resolved.csv",
            "8_3B_Panel2_resolved.csv",
            "8_5B_Panel2_resolved.csv",
        ]
        input_filenames2 = [
            "5_12E_Panel2_GT.csv",
            "7_3A_Panel2_GT.csv",
            "7_5D_Panel2_GT.csv",
            "7_7B_Panel2_GT.csv",
            "8_12B_Panel2_GT.csv",
            "8_1B_Panel2_GT.csv",
            "8_1C_Panel2_GT.csv",
            "8_2A_Panel2_GT.csv",
            "8_3B_Panel2_GT.csv",
            "8_5B_Panel2_GT.csv",
        ]
    else:
        sys.exit("Invalid dataset name in settings['dataset']")

    os.makedirs(
        output_prefix, exist_ok=True
    )  # Make sure directory for output data is created

    xcol = settings["xcol"]
    ycol = settings["ycol"]
    suffix_predicted = settings["suffix_predicted"]
    suffix_annotated = settings["suffix_annotated"]
    selected_core = settings["selected_core"]
    if selected_core >= 0:
        input_filenames = [input_filenames[selected_core]]
        input_filenames2 = [input_filenames2[selected_core]]

    mapping = celltype_info["mapping_predicted"]
    mapping2 = celltype_info["mapping_annotated"]
    labels = celltype_info["labels"]
    cellts = celltype_info["cellts"]

    cores = [s.split(suffix_predicted)[0] for s in input_filenames]
    annotated = []

    for input_filename, input_filename2 in zip(input_filenames, input_filenames2):
        # Read CSV with predicted cell types for this core
        print("Reading predicted cell types from %s..." % input_filename)
        result_core = pd.read_csv(
            input_prefix + input_filename,
            sep="\t",
            encoding=settings["encoding_predicted"],
        )
        generate_final_class(result_core, mapping, labels)
        print("Done.")

        # Read CSV with annotated ground truth cell types for same core
        print("Reading annotated cell types from %s..." % input_filename2)
        annotated_core = pd.read_csv(
            input_prefix2 + input_filename2, encoding=settings["encoding_annotated"]
        )
        generate_final_class(annotated_core, mapping2, labels)
        print("Done.")

        # Some points can be missing in the new dataset, so we will add the
        # predicted label of each point to the closest point in the annotated data
        # by comparing to annotated points binned to buckets of a uniform grid
        print("Transfering predictions to annotated cells...")
        bounds = [
            annotated_core[xcol].min(),
            annotated_core[ycol].min(),
            annotated_core[xcol].max(),
            annotated_core[ycol].max(),
        ]
        grid = Grid(bounds)
        grid_add_points_from_data(grid, annotated_core, xlabel=xcol, ylabel=ycol)
        newcol = [len(cellts) - 1] * len(annotated_core)
        for i in range(0, len(result_core)):
            x = result_core.iloc[i][xcol]
            y = result_core.iloc[i][ycol]
            closest_index = grid_find_closest(grid, (x, y))
            if closest_index >= 0:
                newcol[closest_index] = result_core.iloc[i]["Class"]
        annotated_core["Predicted"] = newcol
        annotated.append(annotated_core)
        print("Done.")

    # Generate common data frame containing all of the cores
    annotated = pd.concat(annotated)

    # Do re-mapping of labels s.t. ambiguous class appears last in the confusion matrix
    pred = (annotated["Predicted"].copy() + len(cellts) - 1) % len(cellts)
    actual = (annotated["Class"].copy() + len(cellts) - 1) % len(cellts)

    # Compute some metrics (accuracy and Cohen's kappa score)
    print("Computing measurements for comparing annotations with predictions")
    accuracy = sum(pred == actual) / len(actual)
    kappa_score = sklearn.metrics.cohen_kappa_score(actual, pred)
    print("Done.")
    print("Accuracy: %f" % accuracy)
    print("Cohen kappa score: %f" % kappa_score)

    # Plot confusion matrix
    output_prefix_cm = output_prefix + (cores[0] if len(cores) == 1 else "all")
    plot_confusion_matrix(actual, pred, cellts, output_prefix_cm + "_")
    plt.xlabel("Predicted cell type")
    plt.ylabel("Annotated cell type")
    plt.title(
        "TMA cores: %s\nAccuracy: %.3f, Cohen's kappa: %.3f"
        % (cores, accuracy, kappa_score),
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(output_prefix_cm + "_cm.png")

    plt.show()


if __name__ == "__main__":
    main()
