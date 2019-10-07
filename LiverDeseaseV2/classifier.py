from __future__ import absolute_import, division, print_function, unicode_literals

#Seed value - Trying to get reproducible results
seed_value=0

import os
os.environ['PYTHONHASHSEED']=str(seed_value)

import random
random.seed(seed_value)

import numpy as np
np.random.seed(seed_value)

import pandas as pd
import tensorflow as tf
tf.random.set_seed(seed_value)

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pdb;

tf.keras.backend.set_floatx('float64')

dataframe = pd.read_csv('liver_dataset.csv')
dataframe.head()
print(dataframe)
pdb.set_trace()
dataframe.Target.replace([1],[0], inplace=True)
dataframe.Target.replace([2],[1], inplace=True)
pdb.set_trace()
# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('Target')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

example_batch = next(iter(df_to_dataset(dataframe)))[0]

# A utility method to create a feature column
# and to transform a batch of data
def demo(feature_column):
  feature_layer = layers.DenseFeatures(feature_column)
  print(feature_layer(example_batch).numpy())


print('Pre processing age==========================>')
age = feature_column.numeric_column("Age")
demo(age)

age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90])
#demo(age_buckets)
#type(age_buckets)

#dataframe.pop('age')
#age_buckets_series = pd.Series(age_buckets)
print('age buckets series====================>')
demo(age_buckets)
age_buckets_df = pd.DataFrame({'Age':age_buckets})
print('age buckets series====================>')

#dataframe.join(age_buckets_column)

print('Pre processing age done==========================>')


#age_buckets_column = age_buckets_column.join(thal_one_hot_column)
#print(age_buckets_column)
#demo(age_buckets_df['age'])
#demo(thal_one_hot_df['thal'])
#dataframe.pop('thal')
dataframe.pop('Gender')

#dataframe['age'] = age_buckets_df['age']
#dataframe['thal'] = thal_one_hot_df['thal']
#print(dataframe)
#print('Pre processing thal done==========================>')
print('Checking for nan===================================>')
print(np.where(np.isnan(dataframe)))
print('Checking for nan===================================>')


print('Dataframe=======================================>')
print(dataframe)
pdb.set_trace()
print('Dataframe=======================================>')

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(dataframe)
normalized = pd.DataFrame(x_scaled, columns = dataframe.columns)
print(normalized)
print('Normalized data====================================>')
pdb.set_trace()

train, test = train_test_split(normalized, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')



batch_size = 5 # A small batch sized is used for demonstration purposes
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)


for feature_batch, label_batch in train_ds.take(1):
  print('Every feature:', list(feature_batch.keys()))
  print('A batch of Direct_Bilirubin:', feature_batch['Direct_Bilirubin'])
  print('A batch of targets:', label_batch )


feature_columns = []

for header in ['Age','Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens','Albumin','Albumin_and_Globulin_Ratio']:

#for header in ['Age','Total_Protiens','Albumin',]:
  feature_columns.append(feature_column.numeric_column(header))

print('Feature columns=================================>')
print(feature_columns)
print('Feature columns=================================>')

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([
  feature_layer,  
  layers.Dropout(0.50, input_shape=(9,)),
  layers.Dense(50, activation='relu'),  
  layers.Dropout(0.50),
  layers.Dense(50, activation='relu'),  
  #layers.Dropout(0.20),
  #layers.Dense(50, activation='relu'),  
  #layers.Dropout(0.20),
  layers.Dense(1, activation='sigmoid')
])

#adam = tf.keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'],
              run_eagerly=True)

model.fit(train_ds,
          validation_data=val_ds,
          epochs=500)
loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)
