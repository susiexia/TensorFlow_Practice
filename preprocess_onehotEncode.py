# %% [markdown]
# implement a oneHotEncoder through sklearn
# use the “country” variable as a categorical variable in a larger dataset that will predict restaurant satisfaction.
# check for total numbers of unique value, ensure dataset wouldnot be too wide

# %%
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder

# %%
ramen_df = pd.read_csv('ramen-ratings.csv')
ramen_df.head(3)

# %%
# uniue values count
country_counts = ramen_df['Country'].value_counts()
country_counts
# we will bucket the rare values into 'other' category because
# There are many unique values, and there is an uneven distribution
# %%
# determine the 'other' category by using a density plot (distribution)
#  identify where the value counts “fall off” and set the threshold within this region.
country_counts.plot.density()

#  bucket any country that appears fewer than 100 times in the dataset as “other”
# %%
# determine which values need to change
relace_countries_lst = list(country_counts[country_counts < 100].index)

# relace 
for cty in relace_countries_lst:
    ramen_df['Country'] = ramen_df['Country'].str.replace(cty, 'Other')
# verify the original df
ramen_df['Country'].value_counts()
# %%
# transpose the variable using one-hot encoding
ohe_model = OneHotEncoder(sparse= False)
encoded_S = ohe_model.fit_transform(ramen_df.Country.values.reshape(-1,1)) # Expected 2D array
# build a new DataFrame 
encoded_df = pd.DataFrame(encoded_S)
encoded_df.columns = ohe_model.get_feature_names(['Country'])
encoded_df.head()
# %%
# merge encode df into original df
new_ramen_df = ramen_df.merge(encoded_df, left_index=True, right_index = True).drop('Country', axis=1)
new_ramen_df.head()

# %%
