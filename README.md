# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
- in this project, we are looking at the [UCI bank marketing dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing), trying to predict if the client will subscribe to a term deposit
- we try two approaches: manually guided Azure Hyperdrive sklearn - hyperparameter optimization vs. an Azure AutoML approach
- the best performing model was the Voting Ensemble selected during the AutoML approach with ~92 percent accuracy

## Scikit-learn Pipeline

**Pipeline Architecture**

- the pipeline progresses step-by-step across:
    - **data preparation**
        - dataset retrieval from    a  given URL with the use of the TabularDataSetFactory
        - data cleaning (creation of dummy values for categorical variables, cleaning dates, etc. )
        - dataset splitting (75:25) for training and testing
    - **model definition**
        - definition of the main LogisticRegression classificator
        - parameterization of the fitting method of the classifier by the two hyperparameters `max_iter` and the regularization parameter `C`
    - **hyper drive preparation**
        - defining an Sklearn estimator for use in the hyperdrive hyperparameter optimization runs
        - definition of the hyperdrive config, setting the sampling methods (`RandomSampling`) for the defined hyperparameters as well as the early stopping policy (`BanditPolicy`)
    - **hyperparameter optimization with hyperdrive**
        - execution on a compute cluster with the hyper drive configs 
        - selecting the best performing Logistic Regression classifier with an accuracy of ~91 percent for the given prediction problem
        - serialization of the best model

**Parameter sampler choice**

- the chose parameter sampler is used to sample from the pre-defined search spaces on the main hyperparameters `C` (for regularization) and `max_iters` (the maximum number of solver iterations to converge)

- the random parameter sampler helps progressing easily through the hyperparamter search space bc. of its non-exhaustive nature, sampling suitable hyperparameters randomly

**Early stopping policy**

- Setting an Early stopping policy on hyperparameter optimization experiments comes with two main advantages: 
    - computations are stopped if there is no more improvement in model accuracy
    - the amount of expended computation resources is limited by stopping after no more improvements can be made 

- in this pipeline a Bandit Policy is defined for early stopping
    - the slack factor determines how much allowed distance from the best run is tolerated for another model training run
    - the evaluation interval determines the frequency of applying the policy during the hyperparameter optimization


## AutoML

The AutoML-approach aims towards automating the complete hand-crafted and thus brittle pipeline described above. It includes data ingestion, feature engineering/learning, model training and hyperparameter selection all in one go.

- Auto ML config allowed for a wide range of models to fitted and checked against the problem at hand. It used Logistic Regression, XGBoost Classifiers, Gradient Boosting  etc. as well as data normalizers, standard scalers, max abs scalers and much more

- Here is an overview over the models tried during the AutoML-experiment:

![AutoML models](/assets/best_models.png)

- Here is an overview over the most important features for the Voting Ensemble model tried during the AutoML-experiment:

![Feature importance for best model](/assets/feature_importance.png)

## Pipeline comparison

- the Logistic Regression accuracy with ~91 percent compares favorably against the Auto ML approach, which selected a Voting Ensemble with ~92 percent accuracy as the best model

- the Auto ML approach however, tried much more extensive dataset transformations, which would have took quite some time to do manually


## Future work
- during the manual data preparation, now data normalization or scaling was done - this might an interesting approach for the manual data science work

- custom dataset balancing capabilities might be interesting, especially in the area of unbalanced datasets, as the dataset appears to be imbalanced:
![imbalanced dataset](/assets/imbalance.png)

- custom [cost-sensitive learning approaches](https://mlr.mlr-org.com/articles/tutorial/cost_sensitive_classif.html) might also be interesting for the problem, helping the model to adapt to avoid costly misclassifications

## Proof of cluster clean up
- here is the screenshot of cluster deletion
![screenshot_dealloc](/assets/proof_b.png)
