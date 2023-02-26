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


# Example

### Build a basic model

```python
df = pd.read_csv('Titanic_train.csv')

df = pd.get_dummies(df,columns=['Sex'])

target = 'Survived'
X = df[['Pclass','Age','Sex_female','Fare']]
X = X.fillna(0)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```

```python
n = 10

### Training
from sklearn import svm
import xgboost as xgb

#clf = RandomForestClassifier(n_estimators=50+(n*2), max_depth=(n*3), random_state=n)
#clf = svm.SVC(C=n,probability=True)
clf = xgb.XGBClassifier(max_depth=n)
clf.fit(X_train,y_train)

### PREDICTIONS
y_pred = clf.predict(X_test)
#y_probas = clf.predict_proba(X_test)
labels = y.unique()
feature_names = list(X_train.columns)

### Save model
save_model(clf,X_train,X_test,y_train,y_test,"Your_user_name",1,"",
           {},
           {},
           {},
           1,
           save = 1
          )
  
```

    
    saved metrics 
    MODEL ID:  Your_user_name-20230226153020
    Tipo_metrica: Clasificacion
    Tipo_modelo: 1
    Algoritmo: XGBClassifier
    umbral: 0.5
    AUC: 0.8466666666666667
    Gini: 0.6933333333333334
    PRAUC: 0.8108876087759457
    F1_score: 0.7467811158798284
    Accuracy: 0.8
    Recall: 0.725
    Precision: 0.7699115044247787
    Fecha: 2023-02-26 15:30:00
    Comentario: 
    
    Modelo, parametros, metricas y stats guardados exitosamente
    
    

### Initiate object model results


```python
model_results = ClassModelResults()
```

### Update all models results


```python
dir_results_files = 'results_files' #Default path
dir_models = 'models' #Default path
```


```python
df_results = model_results.get_model_results(dir_results_files)
```


### Get metrics from all the stored models

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
      <th>PRAUC</th>
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
      <td>Your_user_name-20230226152641</td>
      <td>Clasificacion</td>
      <td>1</td>
      <td>XGBClassifier</td>
      <td>0.5</td>
      <td>0.873905</td>
      <td>0.747810</td>
      <td>0.857136</td>
      <td>0.737327</td>
      <td>0.806780</td>
      <td>0.666667</td>
      <td>0.824742</td>
      <td>2023-02-26 15:26:00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Your_user_name-20230226152648</td>
      <td>Clasificacion</td>
      <td>1</td>
      <td>XGBClassifier</td>
      <td>0.5</td>
      <td>0.846667</td>
      <td>0.693333</td>
      <td>0.810888</td>
      <td>0.746781</td>
      <td>0.800000</td>
      <td>0.725000</td>
      <td>0.769912</td>
      <td>2023-02-26 15:26:00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Your_user_name-21214708</td>
      <td>Clasificacion</td>
      <td>1</td>
      <td>RandomForestClassifier</td>
      <td>0.5</td>
      <td>0.833333</td>
      <td>0.666667</td>
      <td>NaN</td>
      <td>0.776812</td>
      <td>0.786441</td>
      <td>0.778095</td>
      <td>0.780085</td>
      <td>2023-02-21 21:47:00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Your_user_name-20230226153020</td>
      <td>Clasificacion</td>
      <td>1</td>
      <td>XGBClassifier</td>
      <td>0.5</td>
      <td>0.846667</td>
      <td>0.693333</td>
      <td>0.810888</td>
      <td>0.746781</td>
      <td>0.800000</td>
      <td>0.725000</td>
      <td>0.769912</td>
      <td>2023-02-26 15:30:00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Your_user_name-21214715</td>
      <td>Clasificacion</td>
      <td>1</td>
      <td>SVC</td>
      <td>0.5</td>
      <td>0.785476</td>
      <td>0.570952</td>
      <td>NaN</td>
      <td>0.648842</td>
      <td>0.684746</td>
      <td>0.653095</td>
      <td>0.682529</td>
      <td>2023-02-21 21:47:00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Your_user_name-21214718</td>
      <td>Clasificacion</td>
      <td>1</td>
      <td>SVC</td>
      <td>0.5</td>
      <td>0.763810</td>
      <td>0.527619</td>
      <td>NaN</td>
      <td>0.640441</td>
      <td>0.688136</td>
      <td>0.645476</td>
      <td>0.691590</td>
      <td>2023-02-21 21:47:00</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### Get columns

```python

df_results["features_train_cols"]

```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Your_user_name-20230226152648</td>
      <td>Pclass</td>
      <td>Age</td>
      <td>Sex_female</td>
      <td>Fare</td>
      <td>Survived</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Your_user_name-21214718</td>
      <td>Pclass</td>
      <td>Age</td>
      <td>Sex_female</td>
      <td>Fare</td>
      <td>Survived</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Your_user_name-20230226153020</td>
      <td>Pclass</td>
      <td>Age</td>
      <td>Sex_female</td>
      <td>Fare</td>
      <td>Survived</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Your_user_name-21214708</td>
      <td>Pclass</td>
      <td>Age</td>
      <td>Sex_female</td>
      <td>Fare</td>
      <td>Survived</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Your_user_name-21214715</td>
      <td>Pclass</td>
      <td>Age</td>
      <td>Sex_female</td>
      <td>Fare</td>
      <td>Survived</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Your_user_name-20230226152641</td>
      <td>Pclass</td>
      <td>Age</td>
      <td>Sex_female</td>
      <td>Fare</td>
      <td>Survived</td>
    </tr>
  </tbody>
</table>
</div>



### Load specific model

```python
load_model(f"Your_user_name-20230226152641",dir_models=dir_models)
```


    {'chosen_model': XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
                   colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
                   early_stopping_rounds=None, enable_categorical=False,
                   eval_metric=None, gamma=0, gpu_id=-1, grow_policy='depthwise',
                   importance_type=None, interaction_constraints='',
                   learning_rate=0.300000012, max_bin=256, max_cat_to_onehot=4,
                   max_delta_step=0, max_depth=1, max_leaves=0, min_child_weight=1,
                   missing=nan, monotone_constraints='()', n_estimators=100,
                   n_jobs=0, num_parallel_tree=1, predictor='auto', random_state=0,
                   reg_alpha=0, reg_lambda=1, ...)}
                   
### Get best model

```python
model_results.load_best_model('AUC')
```




    {'chosen_model': XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
                   colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
                   early_stopping_rounds=None, enable_categorical=False,
                   eval_metric=None, gamma=0, gpu_id=-1, grow_policy='depthwise',
                   importance_type=None, interaction_constraints='',
                   learning_rate=0.300000012, max_bin=256, max_cat_to_onehot=4,
                   max_delta_step=0, max_depth=1, max_leaves=0, min_child_weight=1,
                   missing=nan, monotone_constraints='()', n_estimators=100,
                   n_jobs=0, num_parallel_tree=1, predictor='auto', random_state=0,
                   reg_alpha=0, reg_lambda=1, ...)}
