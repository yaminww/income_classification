# Income Classification with Ensemble Methods


## Table of Contents

* [Background](##Background)
* [Install](##Install)
* [Structure](##Structure)
* [Code](##Code)
* [Data](##Data)
  * [Dataset](###Dataset)
  * [Source](###Source)
  * [Variables](###Variables)

## Background 
This project is for the Kaggle Competition [Udacity ML Charity Competition](https://www.kaggle.com/competitions/udacity-mlcharity-competition/overview). 
The purpose is to classify if an individual's annual income is above 50,000 usd or not, based on 13 features like age, workclass, etc. 

## Install 

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [XGBoost](https://xgboost.readthedocs.io/en/stable/)

## Structure
	├── model_saved                 # saved trained ML model
	├── prediction                  # predicted labels on dataset `test_census.csv`
	|   |── example_submission.csv  # example on submission to Kaggle
	|   |── ....                    # predictions made by different models
	|── data                        # dataset folder
	|   |── census.csv              # dataset for training and testing
	|   |── test_census.csv         # dataset for making prediction
	├── income_classification.ipynb # Notebook for project
	├── LICENSE
	└── README.md

## Code
Code is provided in the `income_classification.ipynb` notebook file. 

## Data
### Dataset
The dataset `census.csv` is for model training and testing. It consists of approximately 32,000 instances, with each datapoint having 13 features abd 1 label. 

The dataset `test_census.csv` is for making predictions to be submitted to Kaggle. It only consists of 13 features.

### Source 
The dataset is a modified version of the dataset published in the paper ["Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid"](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf) by Ron Kohavi, with the original dataset hosted on [UCI](https://archive.ics.uci.edu/ml/datasets/Census+Income). 

### Variables 
**Features Variables**
- `age`: Age
- `workclass`: Working Class (Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked)
- `education_level`: Level of Education (Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool)
- `education-num`: Number of educational years completed
- `marital-status`: Marital status (Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse)
- `occupation`: Work Occupation (Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces)
- `relationship`: Relationship Status (Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried)
- `race`: Race (White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black)
- `sex`: Sex (Female, Male)
- `capital-gain`: Monetary Capital Gains
- `capital-loss`: Monetary Capital Losses
- `hours-per-week`: Average Hours Per Week Worked
- `native-country`: Native Country (United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands)

**Target Variable**
- `income`: Income Class (<=50K, >50K)
