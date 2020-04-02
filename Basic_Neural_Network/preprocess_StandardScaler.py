# %% [markdown]
# Standardize data for reducing  the overall likelihood that outliers, variables of different units, or skewed distributions

# StandardScaler rescaled variable to a mean od 0, std of 1
# %%
import pandas as pd 
from sklearn.preprocessing import StandardScaler

# %%
hr_df = pd.read_csv('hr_dataset.csv') # from Kaggle
hr_df.head()

# %%
# apply the standardization to the whole df
scaler = StandardScaler()

scaled_hr_data = scaler.fit_transform(hr_df)
# output is ndArray, not a dataframe

# transform into a new DF
scaled_hr_df = pd.DataFrame(scaled_hr_data,
                             columns=hr_df.columns)
scaled_hr_df.head()

# %%
