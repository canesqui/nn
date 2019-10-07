import tensorflow as tf
from tensorflow import keras
import pdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE


#tf.keras.backend.set_floatx('float64')

dataframe = pd.read_csv('liver_dataset.csv')

raw_df = pd.read_csv('liver_dataset.csv')
raw_df.head()
print(raw_df)
print('subset===============================>')
print(raw_df[['Age','Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','Target']])
#pdb.set_trace()

df = raw_df[['Age','Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','Target']]
df.Target.replace([1],[0], inplace=True)
df.Target.replace([2],[1], inplace=True)
print(df)
print('After=============================>')
#pdb.set_trace()

#scaler = preprocessing.StandardScaler()
#norm = scaler.fit_transform(df)
#df = pd.DataFrame(norm, columns=df.columns)

print('After normalization===================================>')
print(df)
print('End normalization===================================>')
pdb.set_trace()

# Use a utility from sklearn to split and shuffle our dataset.
train_df, test_df = train_test_split(df, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.2)
print('After data split============================>')
#pdb.set_trace()
# Form np arrays of labels and features.
train_labels = np.array(train_df.pop('Target'))
val_labels = np.array(val_df.pop('Target'))
test_labels = np.array(test_df.pop('Target'))
print('After pop target============================>')
pdb.set_trace()

train_features = np.array(train_df)
print('Printing train features==================================>')
print (train_features)
print('Printing train features done==================================>')
val_features = np.array(val_df)
test_features = np.array(test_df)
print('After converting features============================>')
#pdb.set_trace()

# Normalize the input features using the sklearn StandardScaler.
# This will set the mean to 0 and standard deviation to 1.
scaler = preprocessing.MinMaxScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

print('After normalization============================>')
pdb.set_trace()

print('Training labels shape:', train_labels.shape)
print('Validation labels shape:', val_labels.shape)
print('Test labels shape:', test_labels.shape)

print('Training features shape:', train_features.shape)
print('Validation features shape:', val_features.shape)
print('Test features shape:', test_features.shape)

print('Before  positive samples metric============================>')
print('Train label shape')
print(train_labels.shape)
print(train_labels)
print('Done printing training labels')
#pdb.set_trace()

result_array = np.unique(train_labels, return_counts=True)


#pdb.set_trace()
neg = result_array[1][0]
pos = result_array[1][1]
total = neg + pos
print('{} positive samples out of {} training samples ({:.2f}% of total)'.format(
    pos, total, 100 * pos / total))
print('Before  positive samples metric============================>')	
#pdb.set_trace()

def make_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dropout(0.25, input_shape=(6,)), 
    tf.keras.layers.Dense(24, activation='linear'),  
    tf.keras.layers.Dense(24, activation='relu'),  
    tf.keras.layers.Dense(1, activation='sigmoid'),
  ])

  metrics = [
      tf.keras.metrics.Accuracy(name='accuracy'),
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc')
  ]
  print('Inside model_make. Before compile===================>')
  model.compile(
      optimizer='adam',
      loss='binary_crossentropy',
      metrics=metrics, run_eagerly=True)
  print('Inside model_make. After compile===================>')
  return model	
  
model = make_model()

EPOCHS = 10
BATCH_SIZE = 32
#train_labels = [float32(x) for x in train_labels]
#val_labels = [float32(x) for x in val_labels]
print('Before model fit========================>')
print(train_features)
print(train_labels)
print(val_features)
print(val_labels)
print('===============================================>')
#pdb.set_trace()

history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(val_features, val_labels))
	
		
epochs = range(EPOCHS)

plt.title('Accuracy')
plt.plot(epochs,  history.history['accuracy'], color='blue', label='Train')
plt.plot(epochs, history.history['val_accuracy'], color='orange', label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

_ = plt.figure()
plt.title('Loss')
plt.plot(epochs, history.history['loss'], color='blue', label='Train')
plt.plot(epochs, history.history['val_loss'], color='orange', label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

_ = plt.figure()
plt.title('False Negatives')
plt.plot(epochs, history.history['fn'], color='blue', label='Train')
plt.plot(epochs, history.history['val_fn'], color='orange', label='Val')
plt.xlabel('Epoch')
plt.ylabel('False Negatives')
plt.legend()	

print('before evaluation========================>')
pdb.set_trace()

results = model.evaluate(test_features, test_labels, verbose=2)
for name, value in zip(model.metrics_names, results):
  print(name, ': ', value)
  
  
  predicted_labels = model.predict(test_features)
cm = confusion_matrix(test_labels, np.round(predicted_labels))

plt.matshow(cm, alpha=0)
plt.title('Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

for (i, j), z in np.ndenumerate(cm):
    plt.text(j, i, str(z), ha='center', va='center')
    
plt.show()

print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
print('Total Fraudulent Transactions: ', np.sum(cm[1]))