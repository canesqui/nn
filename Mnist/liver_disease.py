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


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline


dataset = pd.read_csv('liver_dataset.csv')
dataset.head()
dataset.describe(include='all')
#df = pd.DataFrame({'Gender': ['Male','Female', np.nan]})
#pd.get_dummies(df, prefix=['Gender'], drop_first=True)

dataset = pd.concat([dataset, pd.get_dummies(dataset['Gender'], prefix='Gender')], axis=1)
print(dataset)
dataset.drop(['Gender'], axis=1, inplace=True)
print(dataset)

df = dataset[['Age','Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens','Albumin','Albumin_and_Globulin_Ratio','Gender_Male', 'Gender_Female','Dataset']]
#sns.pairplot(dataset, hue='Dataset')
df.head()
sns.heatmap(df.corr(), annot=True)
#plt.show()

X=df.iloc[:,0:12]
y=df.iloc[:,-1]
print('Printing types==============================>')
tx = type(X)
ty = type(y)
print(tx)
print(ty)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
print('Printing dataset after pre-processing=================>')
print(X)
print(y)
print('Printing dataset shape after pre-processing=================>')
print(X.shape)
print(y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print('Printing Y train================================>')
print(y_train)



from keras import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))

classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))

classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print('Before train===========================>')
print(X_train.to_records(index=False))
print(y_train.to_numpy())
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

eval_model = classifier.evaluate(X_train, y_train)
print('Evaluation==============================>')
print(eval_model)

y_pred=classifier.predict(X_test)
y_pred = (y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)