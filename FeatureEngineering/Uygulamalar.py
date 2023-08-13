############ FETURE ENGINEERING & DATA PREPROCESSING #############
import matplotlib.pyplot as plt

from Fonk import load
import numpy as np
import pandas as pd
df = load()
df.shape
df.head()

# ilk önce bütün değişkenlerimizin harflerini büyük harfe çeviriyoruz

df.columns = [col.upper() for col in df.columns]

df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')

df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"

# Letter Count
df["NEW_NAME_COUNT"] = df["NAME"].str.len()

# Word Count
df["NEW_WORD_COUNT"] = df["NAME"].apply(lambda x : len(str(x).split(" ")))

# Özel Yapıları Yakalamak
# Doktor olanlar var mı diye bakalım
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))

# name title
df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand = False)

# family size
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1

# age pclass
df["NEW_AGE_PCLASS"] = df["AGE"] + df["PCLASS"]

# age level
df.loc[(df["AGE"] < 18), "NEW_AGE_CAT"] = "young"
df.loc[(df["AGE"] >= 18) & (df["AGE"] < 56), "NEW_AGE_CAT"] = "mature"
df.loc[(df["AGE"] >= 56), "NEW_AGE_CAT"] = "senior"

# sex x age
df.loc[(df["SEX"] == "male") & (df["AGE"] <= 21), "NEW_SEX_CAT"] = "youngmale"
df.loc[(df["SEX"] == "male") & (df["AGE"] > 21) & (df["AGE"] <=50), "NEW_SEX_CAT"] = "maturemale"
df.loc[(df["SEX"] == "male") & (df["AGE"] > 50), "NEW_SEX_CAT"] = "seniormale"
df.loc[(df["SEX"] == "female") & (df["AGE"] <= 21), "NEW_SEX_CAT"] = "youngfemale"
df.loc[(df["SEX"] == "female") & (df["AGE"] > 21) & (df["AGE"] <=50), "NEW_SEX_CAT"] = "maturefemale"
df.loc[(df["SEX"] == "female") & (df["AGE"] > 50), "NEW_SEX_CAT"] = "seniorfemale"

df.shape

from Fonk import  grab_col_names, check_outlier,replace_with_tresholds

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

######################## 2. ADIM : Şimdi aykırı değerleri kontrol edebiliriz  ##################################

# Önce aykırı değerleri kontrol ettik
for col in num_cols:
    print(col, check_outlier(df, col))

# aykırı değerleri düzeltelim
for col in num_cols:
    replace_with_tresholds(df, col)

#son halini kontrol edelim bakalım değişkenlerimizdeki aykırı değerler gitmiş mi
for col in num_cols:
    print(col, check_outlier(df, col))

######################### 3. ADIM : Eksik Değer Problemi ##############################

from Fonk import missing_value_table, missing_vs_target

# Eksik değerler
missing_value_table(df)

# CABIN değişkenimiz yerine yeni NEW_CABIN_BOOL (0-1) tpinide değişken eklemiştik bu yüzden bu CABIN değişkeninden kurtulalım

df.drop("CABIN", inplace = True, axis=1)

remove_cols = ["TICKET", "NAME"]
df.drop(remove_cols, inplace=True, axis=1)

# eksik değerleri dolduralım
df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))

# şimsi yaş değişkenindeki eksiklikler gitti yaş değişkenine bağlı olan diğer değişkenleri tekrar oluşturalım

df["NEW_AGE_PCLASS"] = df["AGE"] + df["PCLASS"]

df.loc[(df["AGE"] < 18), "NEW_AGE_CAT"] = "young"
df.loc[(df["AGE"] >= 18) & (df["AGE"] < 56), "NEW_AGE_CAT"] = "mature"
df.loc[(df["AGE"] >= 56), "NEW_AGE_CAT"] = "senior"

df.loc[(df["SEX"] == "male") & (df["AGE"] <= 21), "NEW_SEX_CAT"] = "youngmale"
df.loc[(df["SEX"] == "male") & (df["AGE"] > 21) & (df["AGE"] <=50), "NEW_SEX_CAT"] = "maturemale"
df.loc[(df["SEX"] == "male") & (df["AGE"] > 50), "NEW_SEX_CAT"] = "seniormale"
df.loc[(df["SEX"] == "female") & (df["AGE"] <= 21), "NEW_SEX_CAT"] = "youngfemale"
df.loc[(df["SEX"] == "female") & (df["AGE"] > 21) & (df["AGE"] <=50), "NEW_SEX_CAT"] = "maturefemale"
df.loc[(df["SEX"] == "female") & (df["AGE"] > 50), "NEW_SEX_CAT"] = "seniorfemale"

# kontrol edelim
missing_value_table(df)

# geriye sadece EMBARKED değişkenimiz kaldı

df = df.apply(lambda x: x.fillna(x.mode()[0]) if(x.dtype == "O" and len(x.unique()) <= 10) else x, axis = 0) # tipi object olan ve eşsiz değer sayısı 10 dan küçük olan eksik değerleri mode ile doldurduk


##################### 4. ADIM : Label Encoding ##########################
from EncodingScaling import label_encoder # kendi dosyamdan çağırdım

binary_cols = [col for col in df.columns if df[col].dtype not in ["int64","float64","int32"]
               and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

df.head()

##################### 5. Adım : Rare Encoder #############################

# olası indirgemeleri yaptıktan sonra onehotencoder yapalım diye önce rare encoding yaptık

from EncodingScaling import rare_analyser, rare_encoder

rare_analyser(df, "SURVIVED", cat_cols)

df = rare_encoder(df, 0.01)

df["NEW_TITLE"].value_counts()


########################### 6.ADIM : ONE HOT ENCODING ##########################
from EncodingScaling import one_hot_encoder

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2 ] # 2 ye eşit olanları zaten üstte dönüştürmüştük

df = one_hot_encoder(df, ohe_cols, drop_first=True)

df.head()
df.shape

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

rare_analyser(df, "SURVIVED", cat_cols)

useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]  # kullanışsız sütunlar

# df.drop(useless_cols, axis=1, inplace=True)  # kullanışsız sütunları silebiliriz

########################### 7. ADIM: STANDAR SCALER #######################
# Not : bu problem için ihtiyaç olmayabilir : ihtiyaç durumunda ne yapmamız gerekir ona bakalım
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df[num_cols].head()

################################# 8. ADIM : MODEL ##############################
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

y = df["SURVIVED"]
x = df.drop(["PASSENGERID", "SURVIVED"], axis=1)


X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train,y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

##### YENİ ÜRETTİĞİMİZ DEĞİŞKENLERE Bİ BAKALIM
import seaborn as sns
from matplotlib import pyplot as plt


def plot_importance(model, features, num=len(x), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature' : features.columns})
    plt.figure(figsize=(10,10))
    sns.set(font_Scale = 1)
    sns.barplot(x = 'Value', y='Feature', data = feature_imp.sort_values(by= 'Value',
                                                                         ascending=False)[0:num])

    plt.title('Features')
    plt.tight_layout()
    plt.show()

    if save:
        plt.savefig('importance.png')

plot_importance(rf_model, X_train)

















#########

import pandas as pd
df = pd.DataFrame({"date" : ['2014-12-23','2021-03-13']})

df["date"].str.extract(r"(\d{4})", expand = True)

#################

bdatetime_series = pd.Series(pd.date_range("2015-07-04", periods=4, freq="M"))
df = pd.DataFrame({"date" : bdatetime_series})
df["day"] = df.date.dt.day_name()


