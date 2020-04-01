# %%
import pandas as pd 
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt 

import tensorflow as tf 

# %%
# generate dummy dataset
X, y = make_blobs(n_samples=1000, n_features=2,centers=2, random_state=78)


# %%
# viz with matplotlib by array
plt.scatter(X[:,0], X[:,1], c= y)

# Create a DF for visualization
df = pd.DataFrame(X, columns=['Feature_1','Feature_2'])
df['Target']= y

df.plot.scatter(x='Feature_1',y='Feature_2', c='Target', colormap ='winter')

# %%
# split dataset as train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 78)

# Standardize and Normalized features data for preprocessing
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
# %% [markdown]
# #Build a neural network model, add layers and config it
# %%
# create, initiate a Sequential model
nn_model = tf.keras.models.Sequential()

# add input and first hidden layer
nn_model.add(tf.keras.layers.Dense(units = 1,
                                activation='relu',
                                input_dim= 2))

# add output layer
nn_model.add(tf.keras.layers.Dense( units = 1,
                                activation ='sigmoid'))

# model structure
nn_model.summary()                              
# %%
# compiling, config model for training
nn_model.compile(optimizer='adam', 
                loss='binary_crossentropy',
                metrics =['accuracy'])

# %% [markdown]
# #fit, train this nn_model
# %%
fit_model = nn_model.fit(X_train_scaled, y_train, epochs=100)


# %%
# training history record of loss value and metrics value
history_record = fit_model.history  # it's a dict
history_df = pd.DataFrame(history_record, 
                    index= range(1, len(history_record['loss'])+1))
history_df.head()

# %%
# visualize history
history_df.plot(y='loss')
# %%
history_df.plot(y='accuracy')
# %% [markdown]
# #varified, evaluate this nn_model using the test data

# %%
# Evaluate the model using the test data, only evaluate, not predict
model_loss, model_acc = nn_model.evaluate(X_test_scaled, y_test,
                                            verbose = 2)
print(f"Loss: {model_loss}, Accuracy: {model_acc}")

# %% [markdown]
# #apply model to novel dataset, to predict the classification
# %%
# generate a new dataset
X_new, y_new = make_blobs(n_features=2, n_samples=10, centers=2,random_state=78)

# make a class predictions for input samples
nn_model.predict_classes(X_new)