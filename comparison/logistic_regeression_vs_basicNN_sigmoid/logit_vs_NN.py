# %% [markdown]
# Both are use sigmoid curve to predict the probability (btw 0 and 1),
# of input data belonging to one of two group.

# %%
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# basic neural network
import tensorflow as tf 

# %%
diabetes_df = pd.read_csv('diabetes.csv')
diabetes_df.head()

# %%
# split into X and y
y = diabetes_df.Outcome
X = diabetes_df.drop(columns = ['Outcome'])

# split into train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

# preprocessing and Standardize ONLY for Neural Net.
scaler = StandardScaler()
X_scaler = scaler.fit(X_train)

X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

# %% [markdown]
# # Logistic Regression (sklearn)
# The accuracy score is 0.729
# %%
logit_model = LogisticRegression(solver='lbfgs', max_iter=200) 
#  200 iterations, give model sufficient opportunity to converge on effective weights

logit_model.fit(X_train, y_train)

y_pred = logit_model.predict(X_test)

print(f'Logistic Regression model Accuracy is: {accuracy_score(y_test,y_pred):.3f}')

# %% [markdown]
# # Basic Neural Net (tensorflow.keras)
# The accuracy score is 0.75 with loss: 0.4858
# %%
input_data_dim = len(X_train_scaled[0])
first_hidden = input_data_dim*2

nn_model = tf.keras.models.Sequential()
# input and first layer
nn_model.add(tf.keras.layers.Dense(units = first_hidden, input_dim=input_data_dim,
                                    activation='relu'))
# output layer
nn_model.add(tf.keras.layers.Dense(units = 1, activation ='sigmoid'))
# structure
nn_model.summary()

# %%
# compile 
nn_model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

# train
model_history = nn_model.fit(X_train_scaled,y_train, epochs=50)

# evaluate with test data
model_loss, model_accuracy = nn_model.evaluate(X_test_scaled, y_test,verbose =2)

print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

# %% [markdown]
# # Summary:

# 1. neural networks are prone to overfitting and can be more difficult to train than a straightforward logistic regression model. 

# 2. if you are trying to build a classifier with limited data points (typically fewer than a thousand data points), or if your dataset has only a few features, neural networks may be overcomplicated. 

# 3. logistic regression models are easier to dissect and interpret than their neural network counterparts

# 4. neural networks (and especially deep neural networks) thrive in large datasets. Datasets with thousands of data points, or datasets with complex features, may overwhelm the logistic regression model, while a deep learning model can evaluate every interaction within and across neurons.

# %%
