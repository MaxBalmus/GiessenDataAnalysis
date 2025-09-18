# GiessenDataAnalysis
Code for basic analysis of the Giessen RV pulmonary pressure trace data.

## Setup and installation
Before installation of the ModularCirc package, please setup a virtual environment using either Conda or python virtual environment.

### Conda setup

Install Conda from https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html

Run:
```shell
conda create --name <yourenvname>
conda activate <yourenvname>
```

Proceed to installing GiessenDataAnalysis package.

### Python virtual environment setup

Run `python3 -m venv <yourvenv>`. This creates a virtual environment called venv in your base directory.

Activate the python environment: `source venv/bin/<yourvenv>`.

Proceed to installing GiessenDataAnalysis package. 

## Installation

### pip installation
To install the pip package:

```shell
python -m pip install giessen-data-analysis
```

### Installation from source
Clone the GitHub repo locally:

```shell
git clone https://github.com/MaxBalmus/GiessenDataAnalysis.git
```

After downloading it:
```shell
cd GiessenDataAnalysis
pip install ./
```
This will install the package based on the `pyproject.toml` file specifications.


## Getting started
Instatiate the analysis class using the target csv file path as input:
```python 
ag = analyseGiessen('data/file.csv')
```
Interogate the percentage of data that is covered by an error code:
```python
ag.report_error_percentage()
```
Compute the 1st and 2nd derivatives of the pressure pulse:
```python
ag.compute_derivatives()
```
Results can be found in ```ag.df``` DataFrame.

We can compute pulse values of interest (e.g. systolic, diastolic pressures):
```python
ag.compute_point_of_interest()
```
with the results available in ```ag.points_df``` DataFrame.