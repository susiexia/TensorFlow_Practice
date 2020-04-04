# %% [markdown]
# Once each weak learner is trained, the random forest model predicts the classification based on a consensus of the weak learners.
# In contrast, deep learning models evaluate input data within a single neuron, as well as across multiple neurons and layers.

# Data Resource: https://www.kaggle.com/zaurbegiev/my-dataset#credit_train.csv
# 36,000 rows and 16 feature columns
# classifed, predicted whether or not a loan will or will not be paid provided their current loan status and metrics.
# Conclusion: Random forest predictive accuracy: 0.849;
# Deep_Learning: Loss: 0.39, Accuracy: 0.84
# %%
import pandas as pd 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf 

# %%
loans_df = pd.read_csv('loan_status.csv')
loans_df.head()

# %%
cat_name_list = list(loans_df.dtypes[loans_df.dtypes == 'object'].index)

loans_df[cat_name_list].nunique()

# %%
loans_df['Years_in_current_job'].value_counts()
# all of the categorical values have a substantial number of data points
# no bucketing
# %%
enc = OneHotEncoder(sparse=False)
# Fit and transform the OneHotEncoder using the categorical variable list
encode_df = pd.DataFrame(enc.fit_transform(loans_df[cat_name_list]))

encode_df.columns = enc.get_feature_names(cat_name_list)

# merge and drop 
loans_df = loans_df.merge(encode_df, left_index=True, right_index=True)\
            .drop(cat_name_list, axis=1)

loans_df.head()

# %%
# split into X and y
y = loans_df["Loan_Status_Fully_Paid"]
X = loans_df.drop(columns = ["Loan_Status_Fully_Paid","Loan_Status_Not_Paid"])

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

# standardize
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
# -------------------RandomForest-------------------------------
rf_model = RandomForestClassifier(n_estimators=128, random_state = 78)

rf_model.fit(X_train_scaled, y_train)

y_pred = rf_model.predict(X_test_scaled)

print(f" Random forest predictive accuracy: {accuracy_score(y_test,y_pred):.3f}")

# %%
# -------------------DeepLearning-------------------------------
input_dimension = len(X_train_scaled[0])

nn_model = tf.keras.models.Sequential()
# input and first hidden layer
nn_model.add(tf.keras.layers.Dense(units=24, input_dim =input_dimension,
                                    activation = 'relu'))
# second hidden layer
nn_model.add(tf.keras.layers.Dense(units = 12, activation='relu'))
# output layer
nn_model.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))

# compile with optimizer
nn_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# train, fit
model_history = nn_model.fit(X_train_scaled,y_train, epochs=50)
model_history.history 

# %%
# evaluate with test data
model_loss, model_accuracy = nn_model.evaluate(X_test_scaled,y_test, verbose=2)

print(f"Loss: {model_loss}, \n Accuracy: {model_accuracy}")


# %%
