# A ML Approach to CT Perfusion Imaging

## Abstract
Computed Tomographic Perfusion (CTP) imaging aids the diagnosis and treatment of AIS patients by providing insight into their cerebral hemodynamics. In this project we utilized existing methods of analyzing MRI images with machine learning and applied it to CT scans. The purpose of of our research was to determine if we can compute perfusion parameters, specifically rBV, TTP, MTT
and rBF, through regression analysis.

## Methodologies
We used five machine learning models: K-NearestNeighbors, SVM using a linear model, SVM using a radial basis function model, Adaboost, and Random Forests Decision Trees. Group leave one out cross validation is used on the models to determine the best parameters to use on the data. It is necessary to group the data based on the patient to avoid allowing the data from one patient to be in both the training and test data, decreasing the bias. Leave one out cross validation allows us to use the training data to see which parameters produce the best results to be used on the test set.

## Technologies Used
Python, Numpy, scikit-learn
