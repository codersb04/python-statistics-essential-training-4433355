# %% [markdown]
# # Python Statistics Essential Training

# %% [markdown]
# ## Collecting and Cleaning Data

# %% [markdown]
# ### 01_01 - Loading data
# 
# Using the Ames, Iowa Housing Data https://jse.amstat.org/v19n3/decock.pdf
# 
# 
# Goals:
# 
# 1. Load data from a CSV file using the `pd.read_csv` function.
# 2. Understand how to access and interpret the shape of a DataFrame.
# 3. Apply the `.describe` method to obtain summary statistics for a DataFrame.

# %%
# 01_06
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# make function


def shrink_ints(df):
    mapping = {}
    for col in df.dtypes[df.dtypes == 'int64[pyarrow]'].index:
        max_ = df[col].max()
        min_ = df[col].min()
        if min_ < 0:
            continue
        if max_ < 255:
            mapping[col] = 'uint8[pyarrow]'
        elif max_ < 65_535:
            mapping[col] = 'uint16[pyarrow]'
        elif max_ < 4294967295:
            mapping[col] = 'uint32[pyarrow]'
    return df.astype(mapping)


def clean_housing(df):
    return (df
            .assign(**df.select_dtypes('string').replace('', 'Missing').astype('category'),
                    **{'Garage Yr Blt': df['Garage Yr Blt'].clip(upper=df['Year Built'].max())})
            .pipe(shrink_ints)
            )


url = 'data/ames-housing-dataset.zip'
raw = pd.read_csv(url, engine='pyarrow', dtype_backend='pyarrow')
housing = clean_housing(raw)


# %%
housing

# %%
import numpy as np
import pandas as pd
pd.__version__

# %%
import pandas as pd
url = 'https://github.com/mattharrison/datasets/raw/master/data/ames-housing-dataset.zip'
url = 'data/ames-housing-dataset.zip'
df = pd.read_csv(url, engine='pyarrow', dtype_backend='pyarrow')

# %% [markdown]
# 

# %%
df.shape

# %%
df.head()

# %%
df.describe()

# %% [markdown]
# ### 01_02 - Strings and Categories
# 
# 
# Goals:
# 
# * Understand the data types of columns in a DataFrame using the `.dtypes` attribute.
# * Select and filter categorical columns using the `.select_dtypes` method.
# * Compute and interpret summary statistics for categorical columns using the `.describe` method.
# * Determine the memory usage of string columns in a DataFrame.
# * Convert string columns to the `'category'` data type to save memory.
# 

# %%
df.dtypes

# %%
# Categoricals - Pandas 1.x
df.select_dtypes(object)

# %%
# Categoricals - Pandas 2
df.select_dtypes('string')  # or 'strings[pyarrow]'

# %%
# Categoricals
df.select_dtypes('string').describe().T

# %%
(df
 .select_dtypes('string')
 .memory_usage(deep=True)
 .sum()
)

# %%
(df
 .select_dtypes('string')
 .astype('category')
 .memory_usage(deep=True)
 .sum()
)

# %%
957_287 / 139_033

# %%
df.shape

# %%
# Missing numeric columns (and strings in Pandas 1)
(df
 .isna()
 .mean() 
 .mul(100)
 .pipe(lambda ser: ser[ser > 0])
)

# %%
# Missing string values
(df
 .select_dtypes('string')
 .eq('')
 .mean() 
 .mul(100)
 .pipe(lambda ser: ser[ser > 0])
)

# %%
(df
 .select_dtypes('string')
 .eq('')
 .any(axis='columns')
)


# %%
# deleting
# Notice Alley is NA!
(df
 [~df.select_dtypes('string').eq('').any(axis='columns')]
)

# %%
# Looks like many values are NA
# Missing string values (Encoded as NA)
(df
 .select_dtypes('string')
 .eq('NA')
 .mean() 
 .mul(100)
 .pipe(lambda ser: ser[ser > 0])
)

# %%
(df
 .query('`Pool QC`.isna()')
)

# %%
(df
 .query('`Pool QC` == "NA"')
)

# %%
# Fill in empty string with 'Not Applicable'
(df
 .assign(
     **df.select_dtypes('string').replace('', 'Not Applicable'))
)

# %%
# Examining unique values
# Note the empty string
(df
 .Electrical
 .value_counts()
)

# %%
(df
 .query('Electrical == ""')
)

# %%
# This one was encoded as NA
(df
 ['Fireplace Qu']
 .value_counts()
)

# %%
(df
 ['Bsmt Cond']
 .value_counts()
)

# %%
# Converting to Category
(df
 .assign(
     **df
     .select_dtypes('string')
     .replace('', 'Not Applicable')
     .astype('category')
 )
)

# %%
# Converting to Category
(df
 .assign(
     **df
     .select_dtypes('string')
     .replace('', 'Not Applicable')
     .astype('category')
 )
 .memory_usage(deep=True)
 .sum()
)

# %%
# Converting to Category
(df
 .assign(
     **df
     .select_dtypes('string')
     .replace('', 'Not Applicable')
     #.astype('category')
 )
 .memory_usage(deep=True)
 .sum()
)

# %%


# %%


# %% [markdown]
# ### 01_03 - Cleaning Numbers
# 
# Goals:
# 
# * Select and filter numeric columns using the `.select_dtypes` method.
# * Compute and interpret summary statistics for numeric columns using the `.describe` method.
# * Identify missing values in numeric columns.
# * Display a larger amount of data using options for minimum rows and maximum columns.
# * Utilize the `style` attribute to enhance the display of DataFrames.

# %%
# In Pandas 1.x there would be many numbers here
(df
 .select_dtypes(float)
)

# %%
(df
 .select_dtypes(int)
)

# %%
(df
 .select_dtypes(int)
 .describe()
)

# %%
df.shape

# %%
(df
 .query('`Lot Frontage`.isna()')
)

# %%
# How to see more data
with pd.option_context('display.min_rows', 30, 'display.max_columns', 82):
    display(df
     .query('`Lot Frontage`.isna()')
    )

# %%
df.style.set_sticky?

# %%
with pd.option_context('display.min_rows', 30, 'display.max_columns', 82):
    display(df
     .query('`Lot Frontage`.isna()')
     .style
     .set_sticky(axis='columns') # broken 
     .set_sticky(axis='index')
    )    

# %%
# Examine a column with missing values
(df
 .query('`Garage Yr Blt`.isna()')
 )

# %%
# missing + 2207!!!?
(df
 ['Garage Yr Blt']
 .describe()
)

# %%
# probably a typo!!
with pd.option_context('display.min_rows', 30, 'display.max_columns', 82):  
    display(df.query('`Garage Yr Blt` > 2200'))

# %%
# Any columns with Yr
df.filter?

# %%
(df
 .filter(like='Yr')
)

# %%
# Any columns with Yr > 2023
(df
 .filter(like='Yr')
 .pipe(lambda df_: df_[df_.gt(2023).any(axis='columns')])
)

# %%
# What about "Year" columns?
(df
 .rename(columns=lambda name: name.replace('Yr', 'Year'))
 .filter(like='Year')
 .pipe(lambda df_: df_[df_.gt(2023).any(axis='columns')])
)

# %%
(df
 ['Garage Yr Blt']
 .clip(upper=df['Year Built'].max())
 .value_counts()
 .sort_index()
)

# %%
# Update categories and clip
# Inspect types
(df
 .assign(**df.select_dtypes('string').replace('', 'Missing').astype('category'),
         **{'Garage Yr Blt': df['Garage Yr Blt'].clip(upper=df['Year Built'].max())})
 .dtypes.value_counts()
)

# %%


# %%


# %% [markdown]
# ### 01_04 - Shrinking Numbers
# 
# Goals:
# 
# * Create a function, `shrink_ints`, to automatically convert suitable integer columns to smaller integer types (`uint8`, `uint16`, `uint32`) based on their range of values.
# * Apply the `shrink_ints` function to the DataFrame to reduce memory usage while maintaining data integrity.
# * Create a function, `clean_housing`, that combines the data cleaning steps for string columns, clipping values in the "Garage Yr Blt" column, and shrinking integer columns.
# 

# %%
# continuing where we left off
(df
 .assign(**df.select_dtypes('string').replace('', 'Missing').astype('category'),
         **{'Garage Yr Blt': df['Garage Yr Blt'].clip(upper=df['Year Built'].max())})
 .describe()
)

# %%
for size in [np.uint8, np.uint16, np.uint32, np.uint64]:
    print(np.iinfo(size))

# %%
def shrink_ints(df):
    mapping = {}
    for col in df.dtypes[df.dtypes=='int64[pyarrow]'].index:
        max_ = df[col].max()
        min_ = df[col].min()
        if min_ < 0:
            continue
        if max_ < 255:
            mapping[col] = 'uint8[pyarrow]'
        elif max_ < 65_535:
            mapping[col] = 'uint16[pyarrow]'
        elif max_ <  4294967295:
            mapping[col] = 'uint32[pyarrow]'
    return df.astype(mapping)
            
(df
 .assign(**df.select_dtypes('string').replace('', 'Missing').astype('category'),
         **{'Garage Yr Blt': df['Garage Yr Blt'].clip(upper=df['Year Built'].max())})
 .pipe(shrink_ints)
 .describe()
)

# %%
(df
 .assign(**df.select_dtypes('string').replace('', 'Missing').astype('category'),
         **{'Garage Yr Blt': df['Garage Yr Blt'].clip(upper=df['Year Built'].max())})
 .pipe(shrink_ints)
 .memory_usage(deep=True)
 .sum()
)

# %%
(df
 .memory_usage(deep=True)
 .sum()
)

# %%
1_875_484 / 361_446

# %% [markdown]
# 

# %%
# make function
def shrink_ints(df):
    mapping = {}
    for col in df.dtypes[df.dtypes=='int64[pyarrow]'].index:
        max_ = df[col].max()
        min_ = df[col].min()
        if min_ < 0:
            continue
        if max_ < 255:
            mapping[col] = 'uint8[pyarrow]'
        elif max_ < 65_535:
            mapping[col] = 'uint16[pyarrow]'
        elif max_ <  4294967295:
            mapping[col] = 'uint32[pyarrow]'
    return df.astype(mapping)


def clean_housing(df):
    return (df
     .assign(**df.select_dtypes('string').replace('', 'Missing').astype('category'),
             **{'Garage Yr Blt': df['Garage Yr Blt'].clip(upper=df['Year Built'].max())})
     .pipe(shrink_ints)
    )    

clean_housing(df).dtypes

# %%


# %% [markdown]
# ### 01_05 - Challenge: Clean Ames 
# 
# * Create a cell containing all the imports for this notebook
# * Create a cell with the `clean_housing` and `shrink_ints` functions
# * Add code to load the raw data and create a `housing` variable from calling `clean_housing`
# * Move those cells to the top of the notebook
# * Restart the notebook and make sure that those cells work

# %%
# 01_06
import numpy as np
import pandas as pd

# make function


def shrink_ints(df):
    mapping = {}
    for col in df.dtypes[df.dtypes == 'int64[pyarrow]'].index:
        max_ = df[col].max()
        min_ = df[col].min()
        if min_ < 0:
            continue
        if max_ < 255:
            mapping[col] = 'uint8[pyarrow]'
        elif max_ < 65_535:
            mapping[col] = 'uint16[pyarrow]'
        elif max_ < 4294967295:
            mapping[col] = 'uint32[pyarrow]'
    return df.astype(mapping)


def clean_housing(df):
    return (df
            .assign(**df.select_dtypes('string').replace('', 'Missing').astype('category'),
                    **{'Garage Yr Blt': df['Garage Yr Blt'].clip(upper=df['Year Built'].max())})
            .pipe(shrink_ints)
            )


url = 'data/ames-housing-dataset.zip'
raw = pd.read_csv(url, engine='pyarrow', dtype_backend='pyarrow')
housing = clean_housing(raw)


# %%


# %%


# %%


# %% [markdown]
# ## Exploring & Visualizing

# %% [markdown]
# ### 02_01 - Categorical Exploration
# 
# Goals:
# 
# * Explore a categorical column, such as "MS Zoning," by accessing the column and displaying its unique values.
# * Visualize the value counts of a categorical column using a bar chart.
# * Visualize the value counts of a categorical column using a horizontal bar chart.

# %%
import pandas as pd
url = 'data/ames-housing-dataset.zip'
raw = pd.read_csv(url, engine='pyarrow', dtype_backend='pyarrow')

# make function
def shrink_ints(df):
    mapping = {}
    for col in df.dtypes[df.dtypes=='int64[pyarrow]'].index:
        max_ = df[col].max()
        min_ = df[col].min()
        if min_ < 0:
            continue
        if max_ < 255:
            mapping[col] = 'uint8[pyarrow]'
        elif max_ < 65_535:
            mapping[col] = 'uint16[pyarrow]'
        elif max_ <  4294967295:
            mapping[col] = 'uint32[pyarrow]'
    return df.astype(mapping)


def clean_housing(df):
    return (df
     .assign(**df.select_dtypes('string').replace('', 'Missing').astype('category'),
             **{'Garage Yr Blt': df['Garage Yr Blt'].clip(upper=df['Year Built'].max())})
     .pipe(shrink_ints)
    )    

housing = clean_housing(raw)

# %%
housing.describe()

# %%
# categoricals
(housing
  ['MS Zoning'])

# %%
# categoricals
(housing
  ['MS Zoning']
  .value_counts())

# %%
# categoricals
(housing
  ['MS Zoning']
  .value_counts()
  .plot.bar())

# %%
# categoricals
(housing
  ['MS Zoning']
  .value_counts()
  .plot.barh())

# %%


# %%


# %% [markdown]
# ### 02_02: Histograms and Distributions
# 
# Goals:
# 
# * Obtain descriptive statistics of the "SalePrice" column using the `.describe` method.
# * Visualize the distribution of the "SalePrice" column using a histogram.
# * Customize the histogram by specifying the number of bins using the `bins` parameter.

# %%
# Numerical
(housing
 .SalePrice
 .describe()
)

# %%
# Numerical
(housing
 .SalePrice
 .hist()
)

# %%
# Numerical
(housing
 .SalePrice
 .hist(bins=130)
)

# %%


# %%


# %% [markdown]
# ### 02_03 - Outliers and Z-scores
# 
# Goals:
# 
# * Calculate the Z-score for the "SalePrice" column using the `calc_z` function.
# * Identify outliers based on the Z-score by assigning a boolean column indicating whether the Z-score is greater than or equal to 3 or less than or equal to -3.
# * Identify outliers using the IQR (interquartile range) method by assigning a boolean column indicating whether the values are outside the range of median ± 3 * IQR.
# 

# %%
# outlier with Z-score
def calc_z(df, col):
    mean = df[col].mean() 
    std = df[col].std()
    return (df[col]-mean)/std

(housing
 .pipe(calc_z, col='SalePrice')
)

# %%
(housing
 .assign(z_score=calc_z(housing, col='SalePrice'))
 #.query('z_score.abs() >= 3')
 .query('z_score <= -3')
)

# %%
def calc_iqr_outlier(df, col):
    ser = df[col]
    iqr = ser.quantile(.75) - ser.quantile(.25)
    med = ser.median()
    small_mask = ser < med-iqr*3
    large_mask = ser > med+iqr*3
    return small_mask | large_mask

housing[
calc_iqr_outlier(housing, 'SalePrice')
]

# %%
def calc_iqr_outlier(df, col):
    ser = df[col]
    iqr = ser.quantile(.75) - ser.quantile(.25)
    med = ser.median()
    small_mask = ser < med-iqr*3
    large_mask = ser > med+iqr*3
    return small_mask | large_mask

(housing
 .assign(iqr_outlier=calc_iqr_outlier(housing, col='SalePrice'))
 .query('iqr_outlier')
)

# %%


# %%


# %% [markdown]
# ### 02_04 - Correlations
# 
# Goals:
# 
# * Calculate the Pearson correlation
# * Calculate the Spearman correlation 
# * Color a correlation matrix appropriately

# %%
# Pearson correlation
housing.corr()

# %%
housing.corr(numeric_only=True)

# %%
(housing
 .corr(method='spearman', numeric_only=True)
 .style
 .background_gradient()
)

# %%
(housing
 .corr(method='spearman', numeric_only=True)
 .style
 .background_gradient(cmap='RdBu')
)

# %%
(housing
 .corr(method='spearman', numeric_only=True)
 .style
 .background_gradient(cmap='RdBu', vmin=-1, vmax=1)
)

# %%


# %%


# %% [markdown]
# ### 02_05 - Scatter Plots
# 
# Goals:
# 
# * Create a scatter plot
# * Set transparency
# * Jitter plot values

# %%
(housing
 .plot
 .scatter(x='Year Built', y='Overall Cond')
)

# %%
housing['Year Built'].corr(housing['Overall Cond'], method='spearman')

# %%
(housing
 .plot
 .scatter(x='Year Built', y='Overall Cond', alpha=.1)
)

# %%
# with jitter in y
(housing
 .assign(**{'Overall Cond': housing['Overall Cond'] + np.random.random(len(housing))*.8 -.4})
 .plot
 .scatter(x='Year Built', y='Overall Cond', alpha=.1)
)

# %%
# make function
def jitter(df_, col, amount=.5):
    return (df_
            [col] + np.random.random(len(df_))*amount - (amount/2))
    
(housing
 .assign(#**{'Overall Cond': housing['Overall Cond'] + np.random.random(len(housing))*.8 -.4})
     **{'Overall Cond': jitter(housing, 'Overall Cond', amount=.8)})
 .plot
 .scatter(x='Year Built', y='Overall Cond', alpha=.1)
)

# %%

(housing
 #.assign(**{'Overall Cond': housing['Overall Cond'] + np.random.random(len(housing))*.8 -.4})
 .plot
 .hexbin(x='Year Built', y='Overall Cond', alpha=1, gridsize=18)
)

# %%


# %%


# %% [markdown]
# ### 02_06 - Visualizing Categoricals and Numerical Values
# 
# Goals:
# 
# * Create a box plot of a single column
# * Create a box plot of multiple columns
# * Use the `.pivot` method
# * Use Seaborn to create other distibution plots by category

# %%
# Numerical and categorical
(housing
 #.assign(**{'Overall Cond': housing['Overall Cond'] + np.random.random(len(housing))*.8 -.4})
 .plot
 .box(x='Year Built', y='Overall Cond')
)

# %%
# Make multiple box plots
(housing
 .pivot(columns='Year Built', values='Overall Cond')
 #.apply(lambda ser: ser[~ser.isna()].reset_index(drop=True))
 #.plot.box()
)

# %%
(housing
 .pivot(columns='Year Built', values='Overall Cond')
 .apply(lambda ser: ser[~ser.isna()].reset_index(drop=True))
 .loc[:, [1900, 1920, 1940, 1960, 1980, 2000]]
 .plot.box()
)

# %%
1993 // 10

# %%
# Group by decade
(housing
 .assign(decade=(housing['Year Built']//10 ) * 10)
 .pivot(columns='decade', values='Overall Cond')
 .apply(lambda ser: ser[~ser.isna()].reset_index(drop=True))
 .plot.box()
)

# %%
# or use seaborn
import seaborn as sns

sns.boxplot(data=housing, x='Year Built', y='Overall Cond')

# %%
sns.boxplot?

# %%
sns.boxplot(data=housing, x='Year Built', y='Overall Cond',
            order=[1900, 1920, 1940]
)

# %%
sns.violinplot(data=housing, x='Year Built', y='Overall Cond',
            order=[1900, 1920, 1940]
)

# %%
sns.boxenplot(data=housing, x='Year Built', y='Overall Cond',
            order=[1900, 1920, 1940]
)

# %%


# %%


# %% [markdown]
# ### 02_07 - Comparing Two Categoricals
# 
# Goals:
# 
# 
# * Create a cross-tabulation 
# * Style the cross-tabulation table 
# * Explore the documentation of the `.background_gradient` method of pandas styling.
# * Create a stacked bar plot of a cross-tabulation

# %%
# 2 Categoricals
housing.dtypes[:40]


# %%
# 2 Categoricals - Cross tabulation
(housing
 .groupby(['Overall Qual', 'Bsmt Cond'])
 .size()
 .unstack()
)

# %%
(pd.crosstab(index=housing['Overall Qual'], columns=housing['Bsmt Cond']))

# %%
(pd.crosstab(index=housing['Overall Qual'], columns=housing['Bsmt Cond'])
 .style
 .background_gradient(cmap='viridis', axis=None)  # None is whole dataframe
)

# %%
raw.style.background_gradient?

# %%
# Reorder
# Ex: Excellent
# Gd: Good
# TA: Typical - slight dampness allowed
# Fa: Fair - dampness or some cracking or settling
# Po: Poor - Severe cracking, settling, or wetness
    
(pd.crosstab(index=housing['Overall Qual'], columns=housing['Bsmt Cond'])
 .loc[:, ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'Missing', 'NA']]
 .style
 .background_gradient(cmap='viridis', axis=None)  # None is whole dataframe
)

# %%
# Reorder
# Ex: Excellent
# Gd: Good
# TA: Typical - slight dampness allowed
# Fa: Fair - dampness or some cracking or settling
# Po: Poor - Severe cracking, settling, or wetness
    
(pd.crosstab(index=housing['Overall Qual'], columns=housing['Bsmt Cond'])
 .loc[:, ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'Missing', 'NA']]
 .plot.bar(stacked=True, 
           cmap='viridis')
)

# %%


# %% [markdown]
# ### 02_08 - Challenge: Explore Ames
# * Create a scatter plot of *1st Flr SF* against *SalePrice*

# %%
#02_09 - Solution
(housing
 .plot.scatter(x='SalePrice', y='1st Flr SF'))

# %%
(housing
 #.sample(300)
 .plot.scatter(x='SalePrice', y='1st Flr SF', alpha = .05))


# %%


# %%


# %%


# %%


# %% [markdown]
# ## Linear Regression
# 

# %% [markdown]
# ### 03_01 - Linear Regression
# 
# Goals:
# 
# * Understand how to prepare data for linear regression by selecting the relevant numerical features and the target variable.
# * Learn to split the data into training and testing sets using `train_test_split` 
# * Evaluate the performance of the linear regression model 

# %%
from sklearn import linear_model, model_selection, preprocessing

X = housing.select_dtypes('number').drop(columns='SalePrice')
y = housing.SalePrice

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=42)

# %%
X_train

# %%
y_train

# %%
lr = linear_model.LinearRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)


# %%
X_train.isna().any()

# %%
import pandas as pd
url = 'data/ames-housing-dataset.zip'
raw = pd.read_csv(url, engine='pyarrow', dtype_backend='pyarrow')

# make function
def shrink_ints(df):
    mapping = {}
    for col in df.dtypes[df.dtypes=='int64[pyarrow]'].index:
        max_ = df[col].max()
        min_ = df[col].min()
        if min_ < 0:
            continue
        if max_ < 255:
            mapping[col] = 'uint8[pyarrow]'
        elif max_ < 65_535:
            mapping[col] = 'uint16[pyarrow]'
        elif max_ <  4294967295:
            mapping[col] = 'uint32[pyarrow]'
    return df.astype(mapping)


def clean_housing_no_na(df):
    return (df
     .assign(**df.select_dtypes('string').replace('', 'Missing').astype('category'),
             **{'Garage Yr Blt': df['Garage Yr Blt'].clip(upper=df['Year Built'].max())})
     .pipe(shrink_ints)
     .pipe(lambda df_: df_.assign(**df_.select_dtypes('number').fillna(0)))
    )    
    

housing2 = clean_housing_no_na(raw)

# %%
X = housing2.select_dtypes('number').drop(columns='SalePrice')
y = housing2.SalePrice

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=42)

# %%
lr = linear_model.LinearRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)

# %%


# %%


# %% [markdown]
# ### 03_02 - Interpreting Linear Regression Models

# %%
lr.coef_

# %%
lr.intercept_

# %%
lr.feature_names_in_

# %%
pd.Series(lr.coef_, index=lr.feature_names_in_)

# %%
(pd.Series(lr.coef_, index=lr.feature_names_in_)
 .sort_values()
 .plot.barh())

# %%
(pd.Series(lr.coef_, index=lr.feature_names_in_)
 .pipe(lambda ser: ser[ser.abs() > 100])
 .sort_values()
 .plot.barh())

# %%
(pd.Series(lr.coef_, index=lr.feature_names_in_)
 .pipe(lambda ser: ser[ser.abs() > 100])
 .sort_values()
 .index
)


# %%


# %%


# %% [markdown]
# ### 03_03 - Standardizing Values
# 
# Goals:
# 
# * Understand the concept of standardization 
# * Learn to use the `StandardScaler` class 
# * Evaluate the impact of standardization on the performance of the linear regression model using the coefficient of determination (R-squared) score.
# * Visualize the coefficients of the linear regression model using a horizontal bar plot.

# %%
X = housing2.select_dtypes('number').drop(columns='SalePrice')
y = housing2.SalePrice

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=42)

std = preprocessing.StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.transform(X_test)


# %%
# was .84
lr = linear_model.LinearRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)

# %%
(pd.Series(lr.coef_, index=X.columns)
 .sort_values()
 .plot.barh())

# %%
(pd.Series(lr.coef_, index=X.columns)
 .sort_values()
# .plot.barh()
)

# %%
(pd.Series(lr.coef_, index=X.columns)
 .sort_values()
 .pipe(lambda ser: ser[ser.abs() > 1e8])
 .plot.barh()
)

# %%
(pd.Series(lr.coef_, index=X.columns)
 .sort_values()
 .pipe(lambda ser: ser[ser.abs() > 1e8])
 .index
)

# %%
simple_feats = set(['Kitchen AbvGr', 'Yr Sold', 'Bedroom AbvGr', 'Half Bath',
       'Bsmt Half Bath', 'MS SubClass', 'Full Bath', 'Year Remod/Add',
       'Mo Sold', 'Year Built', 'TotRms AbvGrd', 'Fireplaces', 'Overall Cond',
       'Bsmt Full Bath', 'Garage Cars', 'Overall Qual'])
std_feats = set(['Total Bsmt SF', '2nd Flr SF', '1st Flr SF', 'Low Qual Fin SF',
       'BsmtFin SF 2', 'Gr Liv Area', 'Bsmt Unf SF', 'BsmtFin SF 1'])
print(sorted(simple_feats | std_feats))

# %%
# Look at correlations 
(X
 .assign(SalePrice=y)
 .corr()
 .loc[['SalePrice', '1st Flr SF', '2nd Flr SF', 'Bedroom AbvGr', 'Bsmt Full Bath', 'Bsmt Half Bath', 
       'Bsmt Unf SF', 'BsmtFin SF 1', 'BsmtFin SF 2', 'Fireplaces', 'Full Bath', 
       'Garage Cars', 'Gr Liv Area', 'Half Bath', 'Kitchen AbvGr', 'Low Qual Fin SF',
       'MS SubClass', 'Mo Sold', 'Overall Cond', 'Overall Qual', 'TotRms AbvGrd', 
       'Total Bsmt SF', 'Year Built', 'Year Remod/Add', 'Yr Sold']]
 .style
 .background_gradient(cmap='RdBu', vmin=-1, vmax=1)
 .set_sticky(axis='index') 
)

# %%


# %% [markdown]
# ### 03_04 - Regression with XGBoost
# 
# Goals:
# 
# * Learn to use the XGBoost library (`xgboost`) for regression tasks.
# * Evaluate the performance of the XGBoost model.
# * Explore the importance of features in the XGBoost model using the feature importances.

# %%
X = housing2.select_dtypes('number').drop(columns='SalePrice')
y = housing2.SalePrice

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=42)

std = preprocessing.StandardScaler().set_output(transform='pandas')
X_train = std.fit_transform(X_train)
X_test = std.transform(X_test)


# %%
import xgboost as xgb
# was .84
xg = xgb.XGBRegressor()
xg.fit(X_train, y_train)
xg.score(X_test, y_test)

# %%
pd.Series(xg.feature_importances_, index=X_train.columns).sort_values().plot.barh(figsize=(3,12))

# %%
# Use categories
# (Need to convert pyarrow numbers to pandas 1.x numbers)
X_cat = (housing.assign(**housing.select_dtypes('number').astype('Int64')).drop(columns='SalePrice'))

y_cat = housing.SalePrice
X_cat_train, X_cat_test, y_cat_train, y_cat_test = model_selection.train_test_split(X_cat, y_cat, random_state=42)
xg_cat = xgb.XGBRegressor(enable_categorical=True, tree_method='hist')
xg_cat.fit(X_cat_train, y_cat_train)
xg_cat.score(X_cat_test, y_cat_test)

# %%
pd.Series(xg_cat.feature_importances_, index=xg_cat.feature_names_in_).sort_values().plot.barh(figsize=(3,12))

# %% [markdown]
# ### 03_05 - Challenge: Predict Ames
# * Create a linear regression model using the top 5 features from the (non-categorical) XGBoost model
# * What is the `.score` of the model?

# %%
# 03_06
top5 = (pd.Series(xg.feature_importances_, index= X_train.columns)
 .sort_values()
 .index
 [-5:]
 )
top5

# %%

lr_top5 = linear_model.LinearRegression()
lr_top5.fit(X_train.loc[:, top5], y_train)


# %%
lr_top5.score(X_test.loc[:,top5], y_test)

# %%


# %%


# %%


# %%


# %%


# %% [markdown]
# ## Hypothesis Test

# %% [markdown]
# ### 04_01 - Exploring Data
# 
# Goals:
# 
# * Explore summary statistics by group
# 

# %%
from scipy import stats
housing.Neighborhood.value_counts()

# %%
(housing
 .groupby('Neighborhood')
 .describe())

# %%
(housing
 .groupby('Neighborhood')
 .describe()
 .loc[['CollgCr', 'NAmes'], ['SalePrice']]
)

# %%
(housing
 .groupby('Neighborhood')
 .describe()
 .loc[['CollgCr', 'NAmes'], ['SalePrice']]
 .T
)

# %%


# %%


# %%


# %% [markdown]
# ### 04_02 - Visualizing Distributions
# 
# Goals
# 
# * Make histograms of both distributions
# * Make a cumulative distribution plot

# %%
n_ames = (housing
          .query('Neighborhood == "NAmes"')
          .SalePrice)
college_cr = (housing
          .query('Neighborhood == "CollgCr"')
          .SalePrice)

# %%
ax = n_ames.hist(label='NAmes')
college_cr.hist(ax=ax, label='CollgCr')
ax.legend()

# %%
alpha = .7
ax = n_ames.hist(label='NAmes', alpha=alpha)
college_cr.hist(ax=ax, label='CollgCr', alpha=alpha)
ax.legend()

# %%
(n_ames
 .to_frame()
 .assign(cdf=n_ames.rank(method='average', pct=True))
 .sort_values(by='SalePrice')
 .plot(x='SalePrice', y='cdf', label='NAmes')
)

# %%
def plot_cdf(ser, ax=None, label=''):
    (ser
     .to_frame()
     .assign(cdf=ser.rank(method='average', pct=True))
     .sort_values(by='SalePrice')
     .plot(x='SalePrice', y='cdf', label=label, ax=ax)
    )
    return ser
plot_cdf(n_ames, label='NAmes')

# %%
def plot_cdf(ser, ax=None, label=''):
    (ser
     .to_frame()
     .assign(cdf=ser.rank(method='average', pct=True))
     .sort_values(by='SalePrice')
     .plot(x='SalePrice', y='cdf', label=label, ax=ax)
    )
    return ser
    
fig, ax = plt.subplots(figsize=(8,4))
plot_cdf(n_ames, label='NAmes', ax=ax)
plot_cdf(college_cr, label='CollegeCr', ax=ax)

# %%


# %%


# %%


# %% [markdown]
# ### 04_03 - Running Statistical Tests
# 
# Goals:
# 
# * Use the `scipy.stats` module to run a statistical test

# %%
print(dir(stats))

# %%
stats.ks_2samp?

# %%
ks_statistic, p_value = stats.ks_2samp(n_ames, college_cr)
print(ks_statistic, p_value)

# %%
if p_value > 0.05:
    print('Fail to reject null hypothesis: Same distribution')
else:
    print('Reject null hypothesis: Not from the same distribution')


# %%


# %% [markdown]
# ### 04_04 - Testing for Normality
# 
# Goals:
# 
# * Use the `scipy.stats` module to test for normality
# * Use the `scipy.stats` module to create a probability plot

# %%
# Use the Shapiro-Wilks test
shapiro_stat, p_value = stats.shapiro(n_ames)

# %%
if p_value > 0.05:
    print("The distribution of the series is likely normal (fail to reject H0)")
else:
    print("The distribution of the series is likely not normal (reject H0)")


# %%
p_value

# %%
stats.probplot?

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8,4))
_ = stats.probplot(n_ames, plot=ax)

# %%
alpha = .7
ax = n_ames.hist(label='NAmes', alpha=alpha)
college_cr.hist(ax=ax, label='CollgCr', alpha=alpha)
ax.legend()

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8,4))
_ = stats.probplot(college_cr, plot=ax)

# %%


# %%


# %% [markdown]
# ### 04_05 - Challenge: Checking Square Footage Distributions
# * Is the distribution of *1st Flr SF* from *NAmes* and *CollgCr* the same?

# %%
# 04_06
n_ames_sf = (housing
 .query('Neighborhood == "NAmes"')
 .loc[:,'1st Flr SF']
)

college_cr_sf = (housing
 .query('Neighborhood == "CollgCr"')
 .loc[:,'1st Flr SF']
)

#college_cr_sf

# %%
ax = n_ames_sf.hist()
college_cr_sf.hist(ax = ax, alpha = .7)

# %%
ks, p_value = stats.ks_2samp(n_ames_sf, college_cr_sf)
p_value

# %%


# %%


# %%


# %%



