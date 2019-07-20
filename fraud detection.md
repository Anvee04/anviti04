

```python
import sys
import os
import numpy
import pandas
import matplotlib
import seaborn
import scipy
import sklearn
```


```python
print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(numpy.__version__))
print('Pandas: {}'.format(pandas.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Seaborn: {}'.format(seaborn.__version__))
print('Scipy: {}'.format(scipy.__version__))
print('Sklearn: {}'.format(sklearn.__version__))
```

    Python: 3.7.0 (default, Jun 28 2018, 08:04:48) [MSC v.1912 64 bit (AMD64)]
    Numpy: 1.15.4
    Pandas: 0.23.4
    Matplotlib: 2.2.3
    Seaborn: 0.9.0
    Scipy: 1.1.0
    Sklearn: 0.19.2
    


```python
#import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

```


```python
#load the dataset from the csv file using pandas
dataset = pd.read_csv(r"C:\Users\Anviti\Desktop\creditcard.csv")
```


```python
#explore the dataset
dataset.columns
```


```python
dataset.shape
```


```python
dataset.describe()
```


```python
#taking only 10% of dataset
dataset = dataset.sample(frac = 0.1, random_state = 1)
```


```python
dataset.shape

```


```python
#plot histogram for each parameter

dataset.hist(figsize =(20,20))
plt.show()
```


```python
#finding fraud data in dataset
fraud = dataset[dataset['Class']==1]
valid = dataset[dataset['Class']==0]

outlier_fraction = len(fraud)/float(len(valid))
print(outlier_fraction)

```


```python
#plot heatmap
cormat  = dataset.corr()
fig =  plt.figure(figsize = (9,12))
sns.heatmap(cormat,vmax = .8, square = True)
plt.show()


```


```python
dataset
```


```python
dataset.head()
```


```python
 dataset.corr()
```


```python
#get all the columns from the dataset
dataset.columns.tolist()

```


```python
#filter the columns to remove data we donot want
columns = dataset.columns.tolist()
columns = [c for c in columns if c not in ["Class"]]
target = "Class"
X= dataset[columns]
Y = dataset[target]

print(X.shape)
print(Y.shape)
```


```python
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

#defining random state
state = 1

#define outlier detection method
classifiers = { 
    "Isolation Forest": IsolationForest(max_samples = len(X),contamination = outlier_fraction,random_state = state),
     "Local Outlier Factor": LocalOutlierFactor(n_neighbors = 20,contamination = outlier_fraction)
}


```


```python
n_outliers = len(fraud)

for i,(clf_name,clf) in enumerate(classifiers.items()):
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)

#Reshape the valid to 0 and fraud to 1
    y_pred[y_pred == 1] = 0              
    y_pred[y_pred == -1] = 1
    
    n_errors= (y_pred!= Y).sum()
                
#run the classifiers
    print("classifier: {} {} ".format (clf_name,n_errors))
    print(classification_report(Y,y_pred))
    print(accuracy_score(Y,y_pred))
             
```
