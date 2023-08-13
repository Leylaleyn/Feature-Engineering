############### ENCODING ###############
""" LabelEncoder, kategorik verileri sıralı sayısal değerlere dönüştürmek için kullanılır.
 Yani, bir kategorik sütundaki her farklı değere bir benzersiz sayı atanır.
Bu dönüşüm, sıralanabilen kategorik veriler için uygundur.

OneHotEncoder, kategorik verileri binary (0-1) formatta kodlar.
Bu dönüşüm, kategoriler arasında sıralama olmadığında
 ve her bir kategori arasında eşitlik varsayıldığında uygundur.
Her farklı değer yeni bir sütun olarak temsil edilir ve sadece ilgili hücre 1, diğerleri 0 olur.

Hangi Yöntemi Kullanmalı?

Eğer kategorik verileriniz sıralı ise ve bir tür sıralama içeriyorsa, LabelEncoder kullanabilirsiniz.
Eğer kategorik veriler sıralı değilse veya sıralamalar arasında anlamlı bir ilişki yoksa, OneHotEncoder kullanmak daha uygundur.
"""
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
# Değişkenlerin temsil şekillerinin değiştirilmesi

# Label Encoding : Kategorik değişkenlerin sınıflarını 0-1 şeklinde dönüştürmektedir

#########  LABEL ENCODING & BİNARY ENCODING  #########

from Fonk import grab_col_names

df = load()
df.head()
df["Sex"].head()

le = LabelEncoder()
le.fit_transform(df["Sex"])[0:5]
le.inverse_transform([0,1]) # değerlerin karşılığı neydi hatırlamak istersek

# fonksiyon şeklinde yazalım
def label_encoder(dataframe, binary_col):
    le = LabelEncoder()
    dataframe[binary_col] = le.fit_transform(dataframe[binary_col])
    return dataframe

df = load()

binary_cols = [col for col in df.columns if df[col].dtype not in ["int64","float64"]
               and df[col].nunique() == 2] # len(unique) metodunu kullansaydık eksik değerleri de alacaktı

for col in binary_cols:
    label_encoder(df,col)

df.head()

# daha büyük bir veri setinde inceleyelim
df = load_application_train()
df.shape

binary_cols = [col for col in df.columns if df[col].dtype not in ["int64","float64"]
               and df[col].nunique() == 2] # len(unique) metodunu kullansaydık eksik değerleri de alacaktı

df[binary_cols].head()

for col in binary_cols:
    label_encoder(df,col)

# !! burada EMERGENCYSTATE_MODE değişkenindeki eksik değerleri de doldurdu

# hatırlatma
df = load()
df["Embarked"].value_counts()
df["Embarked"].nunique() # eksik değerleri almadı
len(df["Embarked"].unique()) # eksik değerleri de aldı

################ ONE HOT ENCODING ##############
# dummy değişken tuzağı =
df = load()
df.head()
df["Embarked"].value_counts()

pd.get_dummies(df, columns=["Embarked"], dtype=int).head()


pd.get_dummies(df, columns=["Embarked"],drop_first=True, dtype=int).head() # dummy değişken tuzağına düşmemesi için ilk değişkeni drop ettik

#eksik değerleri için de oluşturmak isteseydik
pd.get_dummies(df, columns=["Embarked"], dummy_na=True, dtype=int).head()

# get dummies ile hem label encoding hem de one hot encoding yapabiliyoruz
pd.get_dummies(df, columns=["Embarked","Sex"], drop_first=True, dtype=int).head()

def one_hot_encoder(dataframe, categorical_cols, drop_first = True):
    dataframe = pd.get_dummies(dataframe, columns= categorical_cols, drop_first=drop_first, dtype=int)
    return dataframe

df = load()

# cat_cols, num_cols, cat_but_car = grab_col_names(df)

ohe_cols = [col for col in df.columns if 10>= df[col].nunique() > 2]

one_hot_encoder(df, ohe_cols).head()


################### RARE ENCODING ###############3
"""
Rare encoding, özellikle kategorik değişkenlerde sık rastlanan nadir sınıfları gruplayarak veya birleştirerek işlem yapma yöntemidir. 
Nadir sınıflar, genellikle veri setinde az sayıda gözlem içeren veya diğer sınıflara göre oldukça az temsil edilen kategorilerdir. 
Bu tür nadir sınıflar, bazen istatistiksel analizlerde veya makine öğrenimi modellerinde sorunlara neden olabilir."""

# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
# 3. Rare encoder yazacağız.

#********
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
#********

df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts()
#incelediğimizde Acedemic degree sınıfının frekansı oldukça az

cat_cols, num_cols, cat_but_car = grab_col_names(df)
print(cat_cols)

def cat_summary(dataframe, col_name, plot= False):
    print(pd.DataFrame({col_name : dataframe[col_name].value_counts(),
                        "Ratio" : 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("--------------------------------------------------")

    if plot:
        sns.countplot(x = dataframe[col_name], data=dataframe)
        plt.show()

# bütün kategorik değişkenler ve sınıf bilgilerine erişebildik
for col in cat_cols:
    cat_summary(df,col)
# Amacımız : Öncelikle kategorik değişkenlere bir dönüşüm işlemi yapmak istiyoruz fakat incelediğimizde kategorik değişkenlerin içerisinde sınıflar var ve bazı sınıflar önemsiz bilgilere sahip
# dolayısıyla önemsiz olan sınıfları encoder edip de datamızı karıştırmamalıyız bu önemsiz sınıfları silmeliyiz bunun için de rare encoder işime


# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.

df["NAME_INCOME_TYPE"].value_counts()
df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()  # 1' e yakın olması kredi ödeyememe gösterirken 1' yakın olmayan tam tersi

# fonksiyonlaştıralım
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({ "COUNT" : dataframe[col].value_counts(),
                             "RATIO" : dataframe[col].value_counts() / len(dataframe),
                             "TARGET_MEAN" : dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df,"TARGET", cat_cols) #elimizdeki bütün kategorik değişkenler için rare analizini elde ettik

# 3. Rare Encoder Yazılması

def rare_encoder(dataframe, rare_perc):  #rare oranı = rare_perc
    temp_df = dataframe.copy()

    # Önce rare columnları seçmekle başlıyoruz
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                   and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis = None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

""" AÇIKLAMA : rare_encoder(dataframe, rare_perc) adlı bir fonksiyon tanımlanmıştır. Bu fonksiyon, iki argüman alır: dataframe (işlem yapılacak veri çerçevesi) ve rare_perc (nadir sınıfların eşik oranı).

temp_df = dataframe.copy() satırı, veri çerçevesinin bir kopyasını oluşturur. İşlem yaparken bu kopya üzerinde değişiklikler yapılacaktır.

rare_columns adlı bir liste oluşturulur. Bu liste, veri çerçevesindeki kategorik sütunlardan ve sütunun nadir sınıflara sahip olup olmadığından oluşur.

for var in rare_columns: döngüsü, nadir sınıflara sahip sütunları tek tek işlemek için kullanılır.

tmp = temp_df[var].value_counts() / len(temp_df) satırı, her sütundaki kategorilerin veri setindeki oranını hesaplar. Bu, nadir sınıfların yüzdesini belirlemek için kullanılır.

rare_labels adlı bir liste oluşturulur. Bu liste, tmp hesaplaması sonucu belirlenen nadir sınıf etiketlerini içerir (yani, nadir sınıf yüzdesi rare_perc oranından küçük olanlar).

temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var]) satırı, nadir sınıflara ait etiketleri "Rare" olarak değiştirir. Yani, nadir sınıfların yerine "Rare" etiketi atanır.

Fonksiyon, işlenmiş veri çerçevesini temp_df olarak döndürür."""

new_df = rare_encoder(df, 0.01)

rare_analyser(new_df,"TARGET", cat_cols)

df["OCCUPATION_TYPE"].value_counts()

################## ÖZELLİK ÖLÇEKLENDİRME(FEATURE SCALING) ###################

# değişkenler arasındaki ölçüm farklılığını gidermek ve kullanılacak olan modellerin değişkenlere eşit şartlar altında yaklaşmasını sağlamak

###########
# StandardScaler = Klasik Standartlaştırma. Ortalamayı çıkarma, standart sapmaya böl. z = (x-u) / s

df = load()
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])
df.head()

############
# RobustScaler : Medyanı çıkar iqr'a böl  (! ortalama ve standart sapma eksik değerden etkilenene metriklerdir, Robust' da median kullanılır StandartScaler'dan farkı budur
# RobustScaler , StandartScaler' a göre aykırı değerlere daha dayanıklıdır

rs = RobustScaler()
df["Age_Robust_Scaler"] = rs.fit_transform(df[["Age"]])
df.describe().T


################
# MinMaxScaler : Verilen 2 değer arasında değişen dönüşümü
mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T

age_cols = [col for col in df.columns if "Age" in col]

# Burada yapmak istediğimiz şey ortaya çıkan değişkenlerin yeni değerlerinde bir değişiklik var mı diye kontrol etmek
def num_summary(dataframe, numerical_col, plot = False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in age_cols:
    num_summary(df,col,plot=True)
# görsellerden anlayacağımız üzere yaş değişkeninin taşıdığı bilgi bozulmadı sadece ifade ediliş tarzı değişti ( scaler  ettigimizde )

# --------------------------
# Numeric to Categorical : Sayısal değişkenleri Kategorik değişkenlere Çevirme
# Binning
# --------------------------

df["Age_qcut"] = pd.qcut(df["Age"], 5,)
df.head()
# qcut metodu bir değişkenin değerlerini küçükten büyüğer doğru sıralar ve çeyrek değerlere göre böler


""" NOT -> TARGET ENCODING : Target Encoding (Hedef Kodlama), 
kategorik bir değişkenin sınıf ortalamalarına göre dönüştürüldüğü bir yöntemdir.
 Her bir kategoriye ait hedef değişkenin ortalama değeri atanır. 
 Ancak, bu yöntemde eğer hedef değişkeni ile kodlanan değişken arasında güçlü bir ilişki varsa 
 (örneğin, hedef değişkeni ile kodlanan değişken aynı ise), model aşırı öğrenmeye eğilimli hale gelebilir. 
 Yani hedef değişkenin veri setindeki varyasyonunu hedeflemek yerine, aşırı öğrenme nedeniyle gürültüyü de öğrenebilir."""


