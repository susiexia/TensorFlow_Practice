# %%
import pandas as pd 
from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt 

import tensorflow as tf 
# %% [markdown]
# # Linear seperable dataset
# %% [markdown]
# A single neuron, single layer model for linear separable dataset, 
# performing a binary classification

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
# # Build a neural network model, add layers and config it
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
# # fit, train this nn_model
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
# # apply model to novel dataset, to predict the classification
# %%
# generate a new dataset
X_new, y_new = make_blobs(n_features=2, n_samples=10, centers=2,random_state=78)
y_new
# %%
# make a class predictions for input samples
nn_model.predict_classes(X_new)

# %% [markdown]
# # Nonlinear dataset

# %% [markdown]
# Use same NN_model to retrain by nonlinear dataset
# %%
# generate dummy nonlinear data
X_moons, y_moons = make_moons(n_samples=1000, noise=0.08, random_state =78)

# transform y_moons to a vertical verctor
y_moons = y_moons.reshape(-1,1)   # -1 means keep orginal row counts

# create a DF to plot the nonlinear dummy data
moons_df = pd.DataFrame(X_moons, columns=["Feature 1", "Feature 2"] )
moons_df['Target']= y_moons

# viz
moons_df.plot.scatter(x='Feature 1', y='Feature 2', 
                        c='Target', colormap ='winter')


# %%
# split
X_moon_train, X_moon_test, y_moon_train, y_moon_test = train_test_split(
                    X_moons, y_moons, random_state =78)

# Standardize
moon_scaler = StandardScaler()
moon_scaler.fit(X_moon_train)

X_moon_train_scaled = moon_scaler.transform(X_moon_train)
X_moon_test_scaled = moon_scaler.transform(X_moon_test)

# %%
# same compiling, same tructure. Only input data is different

#nn_model.compile(optimizer='adam', loss='binary_crossentropy',metrics =['accuracy'])

# %%
# retrain the same sequential model with new nonlinear dataset
moon_fit_model = nn_model.fit(X_moon_train_scaled, y_moon_train, 
                 epochs= 100, shuffle =True)  # whether to shuffle the training data before each epoch

# Epoch 100/100 750/750 [==============================] - 0s 37us/sample - loss: 0.2701 - accuracy: 0.8920

# %%
# evaluate this nn_model using the moon test data
moon_model_loss, moon_model_accuracy = nn_model.evaluate(X_moon_test, y_moon_test)
print(f"Loss: {moon_model_loss}, Accuracy: {moon_model_accuracy}")

# %% [markdown]
# According to the accuracy metric, the basic single-neuron, single-layer neural network model was only able to correctly classify 89% of all data points in the nonlinear training data. Depending on a person’s use case, 89% accuracy could be sufficient for a first-pass model. For example, if we were trying to use a neural network model to separate left-handed people from right-handed people, a model that is correct 89% of the time is very accurate, and guessing incorrectly does not have a huge negative impact.
# However, in many industrial and medical use cases, a model’s classification accuracy must exceed 95% or even 99%. In these cases, we wouldn’t be satisfied with the basic single-neuron, single-layer neural network model, and we would have to design a more robust neural network. In summary, the more complicated and nonlinear the dataset, the more components we’d need to add to a neural network to achieve our desired performance.