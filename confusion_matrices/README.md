# confusion_matrices

Code for generating confusion matrices for cell type predictions


## Basic usage

To generate a confusion matrix from the sample data, activate the `celltype` conda environment from the command prompt or terminal,

    conda activate celltype

and then run

    python plot_confusion.py

For new data, the script `plot_confusion.py` needs to be modified, to specify which CSV files for predictions and annotations to use, and what cell types there are in the data. You can also change the script to for example only generate and plot a confusion matrix for a single CSV file (single TMA core).


## Output

Confusion matrix plots in the folders `output_train` or `output_test`.


## Code style

This project uses the [Black code style](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html). For automatic formatting, the [black code formatter](https://pypi.org/project/black/) can be installed via pip,

    pip install black

and then applied to a source file like this:

    black sourcefile.py

