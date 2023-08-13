import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", None)
pd.set_option('display.float_format', lambda x : '%.3f' % x)
pd.set_option('display.width', 500)

# büyük ölçekli veri seti kullanacağımız zaman
def load_application_train():
    data = pd.read_csv("application_train.csv")
    return data

df = load_application_train()
df.head()

# Küçük ölçekli veri seti kullanacağımız zaman
def load():
    data = pd.read_csv("titanic.csv")
    return data

def outliers_treshold(dataframe, col_name, q1=0.25, q3 =0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outliers_treshold(dataframe, col_name)

    if dataframe[(dataframe[col_name] > up_limit )| (dataframe[col_name] < low_limit)].any(axis = None):
        return True
    else:
        return False


def grab_col_names(dataframe, cat_th = 10, car_th = 20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O" ]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observation :  {dataframe.shape[0]}")
    print(f"Variables :  {dataframe.shape[1]}")
    print(f"cat_cols :  {len(cat_cols)}")
    print(f"num_cols :  {len(num_cols)}")
    print(f"cat_but_car :  {len(cat_but_car)}")
    print(f"num_but_cat :  {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

def grap_outliers(dataframe, col_name, index = False):
    low, up = outliers_treshold(dataframe,col_name)

    if dataframe[((dataframe[col_name] > low) | (dataframe[col_name] > up))].shape[0] > 10:  # burada shape[0] = gözlem sayısı
        print(dataframe[(dataframe[col_name] > low) | (dataframe[col_name])].head())

    else:
        print(dataframe[((dataframe[col_name] > low) | (dataframe[col_name]))])

    if index:
        outier_index = dataframe[((dataframe[col_name] > low) | (dataframe[col_name]))].index
        return  outier_index
def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outliers_treshold(dataframe,col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return  df_without_outliers

def replace_with_tresholds(dataframe, variable):
    low_limit, up_limit = outliers_treshold(dataframe,variable)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outliers_treshold(dataframe, col_name)

    if dataframe[(dataframe[col_name] > up_limit )| (dataframe[col_name] < low_limit)].any(axis = None):
        return True
    else:
        return False

"""from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count = [df.loc[df["NEW_CABIN_BOOL"] == 1, "Suvived"].sum(),
                                               df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],
                                      nobs = [df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                              df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])


"""

def missing_value_table(dataframe, na_name = False):
    na_column = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_column].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_column].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending= False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss','ratio'])
    print(missing_df, end='\n')

    if na_name:
        return na_column



def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + 'NA_FLAG'] = np.where(temp_df[col].isnull(),1,0) # eksiklik gördüğü yere 1 görmediği yere 0 yazar
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("NA")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN" : temp_df.groupby(col)[target].mean(),
                            "COUNT": temp_df.groupby(col)[target].count()}), end="\n\n\n")

## ENCODER

def label_encoder(dataframe, binary_col):
    le = LabelEncoder()
    dataframe[binary_col] = le.fit_transform(dataframe[binary_col])
    return dataframe






