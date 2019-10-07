from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn import preprocessing



#csv_dataframe = pd.read_csv('liver_dataset.csv')
df = pd.read_csv('liver_dataset.csv')

numeric_columns = ['Total_Bilirubin'] #,'Direct_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens','Albumin'] #,'Albumin_and_Globulin_Ratio']

non_numeric_columns = ['Age','Gender']

numeric_data = pd.DataFrame(df, columns=numeric_columns)

non_numeric_data = pd.DataFrame(df, columns=non_numeric_columns)
print(len(numeric_data))
print('Original data')
print(numeric_data)
#import pdb; pdb.set_trace()

plt.plot(numeric_data)
min_max_scaler = preprocessing.StandardScaler()
x_scaled = min_max_scaler.fit_transform(numeric_data)
normalized = pd.DataFrame(x_scaled, columns = numeric_columns)
plt.plot(normalized)
plt.show()
