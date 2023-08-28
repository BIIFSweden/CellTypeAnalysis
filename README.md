# Scripts for cell type classification and analysis


## File structure

- [`qupath_scripts`](qupath_scripts) Groovy scripts used for cell detection and classification in QuPath.
- [`confusion_matrices`](confusion_matrices) Python scripts for plotting confusion matrices.
- [`net_analysis`](net_analysis) Python scripts and Jupyter notebook for performing neighborhood enrichment (NET) analysis.
- [`plot_interactions`](plot_interactions) Python scripts for plotting interactions after NET analysis.


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


## Usage

See respective README.md file or documentation for scripts in each subfolder.


## License

The code is provided under the MIT license (see LICENSE.md).
