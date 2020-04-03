# %% [markdown]

# Dataset Resource: https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset
# Objectives: Tabular Dataset, Classification: identify whether or not a person is likely to depart from the company given his or her current employee profile
# %% [markdown]
# # import, setup and preprocessing
# %%
import pandas as pd 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf 

# %%
attrition_df = pd.read_csv('HR-Employee-Attrition.csv')
attrition_df.head()

# %%
# generate a list of all categorical features name
# DF.dtypes returns a Series
cat_name_list = attrition_df.dtypes[attrition_df.dtypes == 'object'].index.tolist()
cat_name_list   # use this in df as a list of columns name
# %%
# -----------------------encode catigorical variables------------------
# check the number of unique values in each column, perform Bucketing if > 10 unique
attrition_df[cat_name_list].nunique()

# %%
enc = OneHotEncoder(sparse=False)

encode_df = pd.DataFrame(enc.fit_transform(attrition_df[cat_name_list]))
encode_df.columns = enc.get_feature_names(input_features = cat_name_list)

# merge into original DF and drop original cat columns
encoded_attrition_df = attrition_df.merge(encode_df,
                                        left_index = True, 
                                        right_index = True)
encoded_attrition_df = encoded_attrition_df.drop(columns = cat_name_list)
encoded_attrition_df.head()
# %%
# -----------------------2 split steps----------------------------
# split df into independent variables (features X) and depedent variable (y)
y = encoded_attrition_df['Attrition_Yes'].values
X = encoded_attrition_df.drop(columns = ['Attrition_Yes','Attrition_No']).values

# split training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 78)
# %%
# ------------------Standardization X -----------------
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# # Deep Learning Model Design
# %%
# -------------------Create model-----------------------
# define some parameter
number_input_features = len(X_train[0])
hidden_nodes_layer1 =  8
hidden_nodes_layer2 = 5

# create a Sequential model
nn_model = tf.keras.models.Sequential()

# ADD input and first hidden layer
nn_model.add(tf.keras.layers.Dense(units = hidden_nodes_layer1,
                                    input_dim = number_input_features,
                                    activation ='relu'))

# ADD second hidden layer with a few neurons
nn_model.add(tf.keras.layers.Dense(units = hidden_nodes_layer2,
                                    activation = 'relu'))

# ADD output layer, binary classification by predict prob
nn_model.add(tf.keras.layers.Dense(units = 1,
                                    activation = 'sigmoid'))

# check structure of model
nn_model.summary()
# %%
# -------------------compile, config model-----------------------
nn_model.compile(loss = 'binary_crossentropy',
                optimizer = 'adam',
                metrics =['accuracy'])

# %%
# ---------------------train, fit with data-------------------
nn_fit_model = nn_model.fit(X_train_scaled, y_train, epochs=100)

# ---------------------evaluate using test data-----------
model_loss, model_accuracy = nn_model.evaluate(
                                            X_test_scaled, y_test,
                                            verbose = 2)

print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

# %% [markdown]
# the model was able to correctly identify employees 
# who are at risk of attrition approximately 87% of the time. 
# Considering that our input data included more than 30 different variables with more than 1,400 data points, 
# the deep learning model was able to produce a fairly reliable classifier.