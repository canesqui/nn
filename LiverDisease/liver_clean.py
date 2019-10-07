from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import pdb


#csv_dataframe = pd.read_csv('liver_dataset.csv')
df = pd.read_csv('liver_dataset.csv')
df.head()
numeric_columns = ['Age','Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens','Albumin'] #,'Albumin_and_Globulin_Ratio']

#non_numeric_columns = ['Age','Gender']
#non_numeric_columns = ['Age','Gender']

#numeric_data = pd.DataFrame(df, columns=numeric_columns)

#non_numeric_data = pd.DataFrame(df, columns=non_numeric_columns)
#print(len(numeric_data))
#print('Original data')
#print(numeric_data)
#import pdb; pdb.set_trace()



#print('Step 2')
#demo(gender_one_hot)
#print('Gender one hot end=======================>')
#pdb.set_trace()


#print('Bucket')
#numeric_data.assign(Bage=age_buckets)
#print('One hot')
#numeric_data.assign(Gender=gender_one_hot)

#pdb.set_trace()

#min_max_scaler = preprocessing.StandardScaler()
#x_scaled = min_max_scaler.fit_transform(numeric_data)
#normalized = pd.DataFrame(x_scaled, columns = numeric_columns)


#print(len(normalized))
#print('dataframe-------------------------------->')
#print(normalized)
#normalized.head()
#pdb.set_trace()
#normalized['Age'] = pd.DataFrame(df, columns=['Age'])
#print(normalized)
#normalized['Gender'] = pd.DataFrame(df, columns=['Gender'])
#print(normalized)
#normalized['Dataset'] = pd.DataFrame(df, columns=['Dataset'])
#print(normalized)
#import pdb; 
#pdb.set_trace()


#train_test_split comes from scikit library. It is possible to specify
# test_Size or train_size. In this case we are defining the test size as 20%
#of the dataset
train, test = train_test_split(df, test_size=0.2)
print('printing dataset===========================>')
print(train)
print(test)


#validation dataset will be 20% of the train dataset
train, val = train_test_split(train, test_size=0.1)
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

batch_size = 10 # A small batch sized is used for demonstration purposes
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

for feature_batch, label_batch in train_ds.take(1):
  print('Every feature:', list(feature_batch.keys()))
  print('A batch of ages:', feature_batch['Age'])
  print('A batch of targets:', label_batch )



# A utility method to create a feature column
# and to transform a batch of data

# We will use this batch to demonstrate several types of feature columns
example_batch = next(iter(train_ds))[0]

def demo(feature_column):
  feature_layer = layers.DenseFeatures(feature_column)
  print(feature_layer(example_batch).numpy())



#Age,Gender,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio,Dataset


#print('Demo AGE buckets')
#demo(age_buckets)


#print('Gender one hot================================>')
#demo(gender_one_hot)
#print(gender_one_hot)
#pdb.set_trace()

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
#feature_columns.append(age)
# numeric cols

for header in numeric_columns:
  print('Printing header') 
  print(header) 
  feature_columns.append(feature_column.numeric_column(header,dtype=tf.float64))

# bucketized cols

age = feature_column.numeric_column('Age')
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90])

gender  = feature_column.categorical_column_with_vocabulary_list(
      'Gender', ['Male', 'Female'])

gender_one_hot = feature_column.indicator_column(gender)


feature_columns.append(age_buckets)
feature_columns.append(gender_one_hot)

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
print(feature_columns)
demo(feature_columns)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

#batch_size = 32
#train_ds = df_to_dataset(train, batch_size=batch_size)
#val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
#print('TRAIN_DS')
#print(train_ds)
#pdb.set_trace()
#print('VAL_DS')
#print(val_ds)
#test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(12, activation='relu', kernel_initializer='random_normal'),
  layers.Dense(4, activation='relu', kernel_initializer='random_normal'),
  #layers.Dense(24, activation='relu'),
  #layers.Dense(48, activation='relu'),
  #layers.Dense(96, activation='relu'),
  #layers.Dense(192, activation='relu'),
  #layers.Dense(384, activation='relu'),
  #layers.Dense(768, activation='relu'),
  #layers.Dense(1536, activation='relu'),
  layers.Dropout(0.2),
  layers.Dense(1, activation='sigmoid')
])
#pdb.set_trace()
#adam = tf.keras.optimizers.Adam(lr=0.00000000000001)

adam = tf.keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer=adam,
              loss='binary_crossentropy',
              metrics=['accuracy'],
              run_eagerly=False)

model.fit(train_ds,
          validation_data=val_ds,
          epochs=5)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)