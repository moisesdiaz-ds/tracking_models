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
Try it yourself and test it using the Tracking model results - Test notebook.ipynb


# Ejemplo

### Initiate object model results


```python
model_results = ClassModelResults()
```

### Update all models results


```python
dir_results_files = 'results_files'
dir_models = 'models'
```


```python
df_results = model_results.get_model_results(dir_results_files)
len(df_results)
```


### Get metrics

```python
df_results["metrics"]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Tipo_metrica</th>
      <th>Tipo_modelo</th>
      <th>Algoritmo</th>
      <th>umbral</th>
      <th>AUC</th>
      <th>Gini</th>
      <th>F1_score</th>
      <th>Accuracy</th>
      <th>Recall</th>
      <th>Precision</th>
      <th>Fecha</th>
      <th>Comentario</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Your_user_name-19154244</td>
      <td>Clasificacion</td>
      <td>1</td>
      <td>RandomForestClassifier</td>
      <td>0.5</td>
      <td>0.961252</td>
      <td>0.922503</td>
      <td>0.754663</td>
      <td>0.954427</td>
      <td>0.695918</td>
      <td>0.913716</td>
      <td>2023-02-19 15:42:00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Your_user_name-19154302</td>
      <td>Clasificacion</td>
      <td>1</td>
      <td>RandomForestClassifier</td>
      <td>0.5</td>
      <td>0.935442</td>
      <td>0.870884</td>
      <td>0.632842</td>
      <td>0.946880</td>
      <td>0.614626</td>
      <td>0.675028</td>
      <td>2023-02-19 15:43:00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Your_user_name-19154304</td>
      <td>Clasificacion</td>
      <td>1</td>
      <td>RandomForestClassifier</td>
      <td>0.5</td>
      <td>0.923245</td>
      <td>0.846490</td>
      <td>0.774701</td>
      <td>0.958200</td>
      <td>0.720918</td>
      <td>0.915638</td>
      <td>2023-02-19 15:43:00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Your_user_name-19154307</td>
      <td>Clasificacion</td>
      <td>1</td>
      <td>RandomForestClassifier</td>
      <td>0.5</td>
      <td>0.960544</td>
      <td>0.921088</td>
      <td>0.813390</td>
      <td>0.965820</td>
      <td>0.756293</td>
      <td>0.934271</td>
      <td>2023-02-19 15:43:00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Your_user_name-19154311</td>
      <td>Clasificacion</td>
      <td>1</td>
      <td>RandomForestClassifier</td>
      <td>0.5</td>
      <td>0.950667</td>
      <td>0.901333</td>
      <td>0.804382</td>
      <td>0.965893</td>
      <td>0.733333</td>
      <td>0.982428</td>
      <td>2023-02-19 15:43:00</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### Load specific model
```model_results.load_model(f"Your_user_name-19154244")```

RandomForestClassifier(max_depth=9, n_estimators=56, random_state=3)

### Get best model


```model_results.load_best_model()```

RandomForestClassifier(max_depth=45, n_estimators=80, random_state=15)
