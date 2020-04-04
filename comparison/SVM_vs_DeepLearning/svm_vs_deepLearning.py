# %% [markdown]
# compare SVM vs Deep Learning, both are able to deal with non-linear input data, 
# and multiple data types (image and Natural language).

# In many straightforward binary classification problems, SVMs will outperform in two major advantage
# 1. Neural networks and deep learning models will often converge on a local minima. In other words, these models will often focus on a specific trend in the data and could miss the “bigger picture.”

# 2. SVMs are less prone to overfitting because they are trying to maximize the distance, rather than encompass all data within a boundary.

# Data Source: https://www.kaggle.com/raosuny/success-of-bank-telemarketing-data

# Conclusion: SVM model accuracy: 0.873; Deep: Accuracy: 0.87, Loss: 0.36
# %% 
import pandas as pd 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import tensorflow as tf 

# %%
bank_df = pd.read_csv('bank_telemarketing.csv')
bank_df.head()

# %%
# encoding categorical process
cat_name_list = bank_df.dtypes[bank_df.dtypes == 'object'].index.tolist()

# check if any columns have exceed 10 unique values, may need bucketing first 
bank_df[cat_name_list].nunique()

enc = OneHotEncoder(sparse=False)

encode_df = pd.DataFrame(enc.fit_transform(bank_df[cat_name_list]))
encode_df.columns = enc.get_feature_names(cat_name_list)

# merge back
bank_df = bank_df.merge(encode_df, left_index = True, right_index=True)\
                .drop(columns = cat_name_list)

bank_df.head()

# %%
# split into features and target
y = bank_df.Subscribed_yes.values

X = bank_df.drop(columns=['Subscribed_yes','Subscribed_no']).values

# split train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42,
                                                    stratify = y)
# standardization
scaler = StandardScaler()

scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
# -----------------------------SVC(kernel = linear)-----------------------

svc_model = SVC(kernel='linear')
svc_model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = svc_model.predict(X_test_scaled)

print(f" SVM model accuracy: {accuracy_score(y_test,y_pred):.3f}")

# %%
# -----------------------------Deep_Learning-----------------------
input_features = len(X_train_scaled[0])

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(units = 10, input_dim = input_features, 
                                activation = 'relu'))
model.add(tf.keras.layers.Dense(units = 5, activation = 'relu'))

model.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

model.summary()

# %%
# compile the Sequential model together and customize metrics
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# %%
model_history = model.fit(X_train_scaled, y_train, epochs=50)

# %%
model_loss, model_accuracy = model.evaluate(X_test_scaled, y_test)

print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

# %%
