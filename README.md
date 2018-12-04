# Predictive Monitoring with LSTM

## Description
Project branched off from [this](https://github.com/HitLuca/Incremental_predictive_monitoring_of_Business_Processes_with_a_priori_knowledge) repo, where a LSTM based model is tasked with predicting the next timestep of an input business process log.

Differently from the original repo, here we avoid computing custom baselines and instead concentrate on general model tuning, without the need of external servers to inject prior knowledge.

## Project structure

The ```src``` folder contains all the scripts used.

```train_lstm_model.py``` is used to train and evaluate the predictive model.

The ```bpm_lstm``` folder contains all the details regarding model creation and evaluation.

Results are saved into the ```outputs``` folder, along with a tensorboard-compatible logging system.

K-fold cross-validation is used to determine the performance of each model.

## Getting started
This project is intended to be self-contained, so no extra files are required.

### Prerequisites
To install the python environment for this project, refer to the [Pipenv setup guide](https://pipenv.readthedocs.io/en/latest/basics/)

### Running the algorithms
The ```train_lstm_model.py``` script contains all the code necessary to train the predictive models on each log file and evaluate them.
