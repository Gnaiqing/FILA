# FILA: Online Auditing of Machine Learning Model Accuracy under Finite Labelling Budget

![overview](overview.png)

**FILA** is a tool that audit model accuracy of ML classifiers under finite labelling budget. It adaptively sample from unlabeled deployment data, ask human annotators to label the sample, and estimate the F-measure of the model. It leverages the delta method and stratified sampling to derive a sample allocation that approximately minimizes the estimator's variance under finite labeling budget.

### Installation
To install FILA, we suggest using conda environment:
```angular2html
conda create -n FILA python=3.9 
conda activate FILA
python setup.py install
```

### Example Usage

* For a single run of the estimators:

  ```
  python expt.py --dataset_list Amazon-Google --exp_type single
  ```

* Collecting statistics of the estimators:

  ```
  python expt.py --dataset_list Amazon-Google --exp_type stats --n_expts 100
  ```

  The output figures are stored in <code>fig/</code> folder. The name of available datasets can be find in <code>dataset_config.json</code>.

### Modification based on OASIS

The code is developed upon the open-sourced OASIS library, leveraging some of its functionalities such as stratification and initialization.  The code for FILA mainly lie in the following source files:

```
oasis/ass.py: contains code for FILA-Thompson, Neyman-Thompson and Proportional Sampling
expt.py: contains code for conducting experiments
dataset_config.json: contains the configuation for datasets used for experiments
exp_config.json: contains the configuration for adaptive samplers used for experiments
```  