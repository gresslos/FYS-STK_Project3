# FYS-STK_Project3
## Overview
- `CNN.py`                : Contains all steps for data analysis, optimization of the CNN, and final training along with prediction execution.
- `CNN.ipynb`             : An initial attempt to run a complete analysis for the CNN, though the Jupyter kernel crashes. It presents the procedure step-by-step, with titles and some corresponding plots.
- `EnsembleMethods.ipynb` : Contains all data analysis steps for the used ensemble methods (random forests and XGBoost).
- CNN_plots          : Includes the plots produced by `CNN.py`, including optimization plots (heatmaps and validation accuracy evolution), as well as the final training and validation cross-entropy loss and accuracy evolution.
- CNN_outputs        : Includes screenshots of the terminal output produced by `CNN.py`. These screenshots contains information about final model architecture, and accuracy values from the final model.
- Results_Ensemble   : Includes the files and plots produced by `EnsembleMethods.ipynb`.
- Test_runs_ensemble : Contains a debug run of the data analysis for the ensemble methods on the Iris set (`IrisTest.ipynb`) and the results produced by it(Results_Iris).

  

## Functions
### CNN (found in `CNN.py` and `CNN.ipynb`)
- **Input Variables**:
  - `eta`: Learning rate for the optimizer.
  - `lmbd`: L2 regularization parameter for kernel weights.
  - `n_filters`: Number of filters for the convolutional layers.
  - `n_conv`: Size of the convolutional kernel (height and width).
  - `n_pool`: Pooling size (height and width) for max pooling layers.
  - `n_neurons`: Number of neurons in the dense layers.
  - `act_func`: Activation function for all layers (e.g., ReLU, LReLU).
  - `optimizer`: Optimizer function (e.g., Adam).
  - `extra_dense_layers`: (Optional) Number of additional dense layers to add after the initial one.

  The function builds a CNN using the Keras Sequential API. It consists of multiple convolutional and max-pooling layers, followed by flattening and fully connected (dense) layers. The function also supports additional dense layers specified by the `extra_dense_layers` parameter. Finally, a softmax output layer is added for multi-class classification.

  The model is compiled with a categorical cross-entropy loss function and accuracy as the evaluation metric. The optimizer's learning rate is set according to the `eta` parameter.

- **Returns**:
  - A compiled Keras `Sequential` model ready for training.

- **Notes**:
  - The function uses L2 regularization (`regularizers.l2(lmbd)`) for all convolutional and dense layers.
  - The output layer uses the `softmax` activation function for multi-class classification.
  - For more details on the Keras `Sequential` model, refer to the [official documentation](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential).


### GridSearchXGBoost (found in `EnsembleMethods.ipynb` and `IrisTest.ipynb`)
- **Input Variables**:
  - `x_train`, `y_train`: Training inputs and targets
  - `n_folds`: number of folds for cross validation
  - `eta_vals`: numpy array containing the values of the learning rate to test 
  - `lambda_vals`: numpy array containing the values of the L2 regularization to test
  
  Performs an `n-folds` cross validation analysis on a set of training data (`x_train`, `y_train`),
  utilizing an `XGBClassifier` with 100 iterations of max_depth 1. Does so over a grid of [`eta_vals`, `lambda_vals`]
  of learning rates and L2 regs. 
- **Saves**:
  - `results_csv`: .csv file containing the mean validation accuracies for all combinations of Learning Rate and L2 reg.
  - `best_settings_XGB`: .txt file containing the optimal hyperparameter settings, their mean validation accuracy and their position on the parameter grid

### GridSearchRF (found in `EnsembleMethods.ipynb` and `IrisTest.ipynb`)
- **Input Variables**:
  - `x_train`, `y_train`: Training inputs and targets
  - `n_folds`: number of folds for cross validation
  - `depth_vals`: numpy array containing the values of max_depth to test 
  
  Performs an `n-folds` cross validation analysis on a set of training data (`x_train`, `y_train`),
  utilizing a `RandomForestClassifier` with `entropy` as the splitting criterion. Does so over a grid of [`depth_vals`]
  of max_depth. 
- **Saves**:
  - `results_csv`: .csv file containing the mean validation accuracies for all combinations of max_depth and n_estimators.
  - `best_settings_RF`: .txt file containing the optimal hyperparameter settings, their mean validation accuracy and their position on the parameter grid

### LossAccuracyOverRounds (found in `EnsembleMethods.ipynb` and `IrisTest.ipynb`)
- **Input Variables**:
  - `x_train`, `y_train`: Training inputs and targets
  - `x_test`, `y_test`: Testing inputs and targets
  - `eta`: learning_rate parameter for `XGBClassifier` 
  - `lamb`: reg_lambda parameter for `XGBClassifier`
  - `maxrounds`: maximum number of boosting rounds to perform
  
  Fits an `XGBClassifier` model with `maxrounds` iterations, `eta` learning rate and `lambda`L2 regularization to 
  to a set of training data (`x_train`, `y_train`). Afterwards, performs predictions on both the training and the
  testing (`x_test`, `y_test`) data utilizing only the `i` first iterations, with `i` going from 1 to `maxrounds`,
  saving the accuracy scores and cross entropy values obtained for those predictions
- **Saves**:
  - `best_iteration_XGB`: .txt file containing the iteration number with the highest test accuracy (along with the corresponding accuracy), as well as the training and testing accuracies after all the `maxrounds` boosting steps
  - `best_train_XGB`, `final_train_XGB`: .npy files containing the predictions on the training set both at the best and last iterations
  - `best_test_XGB`, `final_test_XGB`: .npy files containing the predictions on the test set both at the best and last iterations
  - `train_accs_XGB`, `train_errors_XGB`: .npy files containing the arrays with the accuracy and entropy values obtained on the training set
  - `test_accs_XGB`, `test_errors_XGB`: .npy files containing the arrays with the accuracy and entropy values obtained on the testing set

### LossAccuracyOverEstimators (found in `EnsembleMethods.ipynb` and `IrisTest.ipynb`)
- **Input Variables**:
  - `x_train`, `y_train`: Training inputs and targets
  - `x_test`, `y_test`: Testing inputs and targets
  - `depth`: max_depth parameter for `RandomForestClassifier` 
  - `estimators`: n_estimators parameter for `RandomForestClassifier`
  
  Fits a `RandomForestClassifier` model with `estimators` estimators of max depth `depth` to a set of training data (`x_train`, `y_train`). 
  Afterwards, performs predictions on both the training and the testing (`x_test`, `y_test`) data utilizing only the `i` first trees in the forest, 
  with `i` going from 1 to `estimators`, saving the accuracy scores and cross entropy values obtained for those predictions
- **Saves**:
  - `best_estimators_RF`: .txt file containing the number of trees with the highest test accuracy (along with the corresponding accuracy), as well as the training and testing accuracies using the total number of trees (`estimators`) 
  - `best_train_XGB`, `final_train_XGB`: .npy files containing the predictions on the training set both at the best and last number of trees
  - `best_test_XGB`, `final_test_XGB`: .npy files containing the predictions on the test set both at the best and last number of trees
  - `train_accs_RF`, `train_errors_RF`: .npy files containing the arrays with the accuracy and entropy values obtained on the training set
  - `test_accs_RF`, `test_errors_RF`: .npy files containing the arrays with the accuracy and entropy values obtained on the testing set