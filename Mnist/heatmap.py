import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline


dataset = pd.read_csv('liver_dataset.csv')
dataset.head(2)
dataset.describe(include='all')
#df = pd.DataFrame({'Gender': ['Male','Female', np.nan]})
#pd.get_dummies(df, prefix=['Gender'], drop_first=True)

dataset = pd.concat([dataset, pd.get_dummies(dataset['Gender'], prefix='Gender')], axis=1)
print(dataset)
dataset.drop(['Gender'], axis=1, inplace=True)
print(dataset)
#sns.pairplot(dataset, hue='Dataset')
sns.heatmap(dataset.corr(), annot=True)
plt.show()
#uniform_data = np.random.rand(10, 12)
#sns.heatmap(uniform_data)
#plt.show()