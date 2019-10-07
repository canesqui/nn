from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

import tensorflow as tf

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


min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(numeric_data)
normalized = pd.DataFrame(x_scaled, columns = numeric_columns)


print(len(normalized))
print('dataframe-------------------------------->')
print(normalized)
normalized.head()
#pdb.set_trace()
normalized['Age'] = pd.DataFrame(df, columns=['Age'])
print(normalized)
normalized['Gender'] = pd.DataFrame(df, columns=['Gender'])
print(normalized)
normalized['Dataset'] = pd.DataFrame(df, columns=['Dataset'])
print(normalized)
#import pdb; 
#pdb.set_trace()


#train_test_split comes from scikit library. It is possible to specify
# test_Size or train_size. In this case we are defining the test size as 20%
#of the dataset
train, test = train_test_split(normalized, test_size=0.2)

#validation dataset will be 20% of the train dataset
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('Dataset')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

batch_size = 32 # A small batch sized is used for demonstration purposes
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

for feature_batch, label_batch in train_ds.take(1):
  print('Every feature:', list(feature_batch.keys()))
  print('A batch of ages:', feature_batch['Age'])
  print('A batch of targets:', label_batch )

# We will use this batch to demonstrate several types of feature columns
example_batch = next(iter(train_ds))[0]

# A utility method to create a feature column
# and to transform a batch of data
def demo(feature_column):
  feature_layer = layers.DenseFeatures(feature_column)
  print(feature_layer(example_batch).numpy())

#Age,Gender,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio,Dataset

age = feature_column.numeric_column('Age')
print("Demo AGE")
demo(age)

age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90])
print('Demo AGE buckets')
demo(age_buckets)

gender  = feature_column.categorical_column_with_vocabulary_list(
      'Gender', ['Male', 'Female'])

gender_one_hot = feature_column.indicator_column(gender)
print('Gender one hot')
demo(gender_one_hot)

# Notice the input to the embedding column is the categorical column
# we previously created
#thal_embedding = feature_column.embedding_column(thal, dimension=8)
#demo(thal_embedding)

#thal_hashed = feature_column.categorical_column_with_hash_bucket(
#     'thal', hash_bucket_size=1000)
#demo(feature_column.indicator_column(thal_hashed))


#crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
#demo(feature_column.indicator_column(crossed_feature))

feature_columns = []

#Age,Gender,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio,Dataset
#feature_columns.append(gender_one_hot)
#feature_columns.append(age_buckets)
# numeric cols

for header in numeric_columns:
  print('Printing header') 
  print(header) 
  feature_columns.append(feature_column.numeric_column(header))

# bucketized cols
#age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
#feature_columns.append(age_buckets)
#feature_columns.append(gender_one_hot)

#min_max_scaler = preprocessing.MinMaxScaler()
#feature_values = min_max_scaler.fit_transform(feature_columns.values)
#feature_columns.values = feature_values
#dataframe = pd.DataFrame(x_scaled, columns = csv_dataframe.columns)


# embedding cols
#thal_embedding = feature_column.embedding_column(thal, dimension=8)
#feature_columns.append(thal_embedding)

# crossed cols
#crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
#crossed_feature = feature_column.indicator_column(crossed_feature)
#feature_columns.append(crossed_feature)

print('Feature columns------------------------------------------>')
#print(feature_columns)
demo(feature_columns)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
print('VAL_DS')
print(val_ds)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(1, activation='relu'),  
  #layers.Dense(512, activation='relu'),
  #layers.Dense(512, activation='relu'),
  #layers.Dense(512, activation='relu'),
  #layers.Dense(512, activation='relu'),
  #layers.Dense(512, activation='relu'),
  layers.Dropout(0.05),
  layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'],
              run_eagerly=True)

model.fit(train_ds,
          validation_data=val_ds,
          epochs=15)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)
