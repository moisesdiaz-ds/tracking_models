# Tracking models
Simple framework to save and track your model versions performance on the exploration and training phase

# How it works
The framework only uses the save_model function and the ClassModelResults class in order to effectively and easily track your model versions 

## save_model function
The save_model function locally saves:
- A pickle file of your model
- The hyperparameters of the algorithim you used
- The perfomance metrics
- Some stats that characterize your model data
- The features and target variable
- The mean X_train feature values for tracking purposes
- Any other metric you want to add

## ClassModelResults class
This class loads all the resutls from saved from the save_model function and summarize it in a dictionary with the following dataframes:
- params: The hyperparameters of the algorithim every model was built
- metrics: The perfomance metrics of every model
- stats: Some stats that characterize every model data
- features_train_cols: The features and target variable of every model
- features_train_mean: The mean X_train feature values of every model

NOTE: you can also specify that you only want to load the n last results, using `ClassModelResults(last_results=n)`

# Try it yourself
Try it yourself and test it using the Tracking model results - Test notebook
