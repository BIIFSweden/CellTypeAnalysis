# confusion_matrices

Code for generating confusion matrices for cell type predictions


## Python installation (via Anaconda and pip):

1. Install the Anaconda (Miniconda) package manager for Python 3.9 from [here](https://docs.conda.io/en/latest/miniconda.html). On Linux, you can also install it like this:
```
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  
    sh Miniconda3-latest-Linux-x86_64.sh
```
2. Create a new virtual environment (celltype) for the application, from the terminal or Anaconda command line:
```
    conda create --name celltype python=3.9
```
3. Activate the virtual environment and install the required Python dependendecies (via pip):
```
    conda activate celltype
    pip install -r requirements.txt
```


## Basic usage

To generate a confusion matrix from the sample data, run

    python plot_confusion.py

For new data, the script `plot_confusion.py` needs to be modified, to specify which CSV files for predictions and annotations to use, and what cell types there are in the data. You can also change the script to for example only generate and plot a confusion matrix for a single CSV file (single TMA core).


## Output

Confusion matrix plots in the folders `output_train` or `output_test`.


## Code style

This project uses the [Black code style](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html). For automatic formatting, the [black code formatter](https://pypi.org/project/black/) can be installed via pip,

    pip install black

and then applied to a source file like this:

    black sourcefile.py


## License

The code is provided under the MIT license (see LICENSE.md).
