# GiessenDataAnalysis
Code for basic analysis of the Giessen RV pulmonary pressure trace data.


## Installation
Clone the repository:
```shell
git clone git@github.com:MaxBalmus/GiessenDataAnalysis.git
```
If no environment exists, create a new one:
```shell
python -m venv myenv
```
Then install the packege with pip:

```shell
pip install .
```

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