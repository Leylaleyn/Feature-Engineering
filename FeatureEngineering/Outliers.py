# Eğer datanız kötüyse kullanacağımız toollar kullanışsızdır ! #
# Feature Engineering(Özellik Mühendisliği) : Ham veriden değişken üretmek.
# Veri Ön işleme :  Çalışmalar öncesi verinin uygun hale getirilmesi sürecidir.
# özellik mühendisliği veri ön işelemenin alt kümesidir.

################### OUTLIERS ( AYKIRI DEĞERLER ) #########################

""" NOT : Aykırı değerler neye göre belirlenir:
1) Sektör bilgisi,
2) Standart Sapma Yaklaşımı,
3) Z-Skoru Yaklaşımı,
4) Booxplot(interquartile range- IQR) Yöntemi  """

########### AYKIRI DEĞERLER YAKALAMA ############

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

df = load()
df.head()

sns.boxplot(x = df["Age"])
plt.show()

# yaş değişkenindeki aykırı değeri yakalamak istersek

q1 = df["Age"].quantile(0.25) # yüzde 25 lik değere bakıyoruz
q3 = df["Age"].quantile(0.75)

iqr = q3 -q1
up = q3 + 1.5 * iqr # yaş değeri için üst değer
low = q1 - 1.5 * iqr # yaş değeri için alt değer

df[(df["Age"] < low) | (df["Age"] > up)] #aykırı değerler

df[(df["Age"] < low) | (df["Age"] > up)].index #aykırı değerlerin indexleri

# Aykırı Değer var mı yok mu?

df[(df["Age"] < low) | (df["Age"] > up)].any(axis=None)

df[df["Age"] < low].any(axis=None)

# NELER YAPTIK :
# 1. Eşik değeri belirledi
# 2. Aykırılara eriştik
# 3. Hızlıca aykırı değer var mı yok mu diye sorduk

###### FONKSİYONLAŞTIRMA #######

def outliers_treshold(dataframe, col_name, q1=0.25, q3 =0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outliers_treshold(df,"Age")
outliers_treshold(df,"Fare")

df.head()
"""
for col in "Age","Fare":
    low, up = outliers_treshold(df,col)
    print(df[(df[col] > low) | (df[col] < up)].head()) """

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outliers_treshold(dataframe, col_name)

    if dataframe[(dataframe[col_name] > up_limit )| (dataframe[col_name] < low_limit)].any(axis = None):
        return True
    else:
        return False

check_outlier(df,"Survived")

############# grab_col_names #########

dff = load_application_train()
dff.head()

def grab_col_names(dataframe, cat_th = 10, car_th = 20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat  kardinal değişkenlerin isimlerini verir
    Not : Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir
    Parameters
    ----------
    dataframe
       değişken isimleri alınmak istenen dataframe'dir.

    cat_th : int, float
        numerik fakat kategorik olan değişkenler için sınıf eeşik değeri
    car_th : int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols : list
               Kategorik değişken listesi
        num_cols : list
               Numeric Değişken listesi
        cat_but_car : list
               Kategorik görünümlü kardinal değişken liste

    Notes
    ------
    cat_cols + num_cols + cat_but_car =  toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde
    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O" ]
    num_but_cat = [col for col in dataframe.columns if df[col].nunique() < cat_th and dataframe[col].dtypes != "O"]

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

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# PassengerId istemıyoruz
num_cols = [col for col in num_cols if col not in "PassengerId"]

for col in num_cols:
    print(col, check_outlier(df,col))

# diğer veri seti için bakalım
cat_cols, num_cols, cat_but_car = grab_col_names(dff)

for col in num_cols:
    print(col, check_outlier(dff,col))


################### AYKIRI DEĞERLERE ERİŞME ####################

def grap_outliers(dataframe, col_name, index = False):
    low, up = outliers_treshold(dataframe,col_name)

    if dataframe[((dataframe[col_name] > low) | (dataframe[col_name] > up))].shape[0] > 10:  # burada shape[0] = gözlem sayısı
        print(dataframe[(dataframe[col_name] > low) | (dataframe[col_name])].head())

    else:
        print(dataframe[((dataframe[col_name] > low) | (dataframe[col_name]))])

    if index:
        outier_index = dataframe[((dataframe[col_name] > low) | (dataframe[col_name]))].index
        return  outier_index


grap_outliers(df,"Age")
age_index = grap_outliers(df,"Age",True)

################### AYKIRI DEĞER PROBLEMİNİ ÇÖZME #####################

# SİLME

low, up = outliers_treshold(df,"Fare")
df.shape

df[~((df["Fare"] < low) | (df["Fare"] > up))].shape #shape ine baktığımız zaman aykırı değerlerin silinmiş oluğunu gördük

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outliers_treshold(dataframe,col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return  df_without_outliers

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

for col in num_cols:
    new_df = remove_outlier(df,col)

df.shape[0] - new_df.shape[0]

# Dikkat : aykırı değerlerden dolayı silme işlemi yaptığımz zaman diğer değişkenleri de kaybetmiş oluyoruz

### Baskılama Yöntemi ( re-assignment with tresholds)

low, up = outliers_treshold(df,"Fare")

df[(df["Fare"] < low) | (df["Fare"] > up)]["Fare"] #Aykırı değer olan Fare'lar

# df.loc[((df["Fare"] < low) | (df["Fare"] > up)),"Fare"] şeklinde de yazabilirdik

df.loc[df["Fare"] > up, "Fare"] = up # burada üst sınıfa göre aykırı olan değerleri üst sınıfa eşitleyerek aykırılık problemini çözmeye çalıştık
df.loc[df["Fare"] > low, "Fare"] = low

# bu işlemi fonksiyonlaştıralım

def replace_with_tresholds(dataframe, variable):
    low_limit, up_limit = outliers_treshold(dataframe,variable)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit

# Şimdiİşlemlerimiz Datasetimize sırasıyla uygulayalım

#datasetimizi yükleyelim
df = load()
df.shape

# num_cols u alalım
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

# outliers var mı yok mu kontrol edelim
for col in num_cols:
    print(col, check_outlier(df,col))

# replace trashold'u getirelim ve her bir aykırı değerleri değiştirelim

for col in num_cols:
    replace_with_tresholds(df, col)

# şimdi tekrar treshold olup olmadığını kontrol edelim

for col in num_cols:
    print(col, check_outlier(df,col))

#### RECAP ####
df = load()
outliers_treshold(df,"Age") # aykırı değeri saptadık
check_outlier(df,"Age")  # aykırı değer var mı yok mu sorusunu sorduk
grap_outliers(df,"Age", index=True) # outlier ları getirttik
# tedavi edelim dedik
remove_outlier(df,"Age").shape # silerek tedavi edebiliriz ama bu değişken sayısını azaltıyor

replace_with_tresholds(df,"Age") # yerine koyma ile tedavi ettik
check_outlier(df,"Age") #en son kontrol ettik


################ ÇOK DEĞİŞKENLİ AYKIRI DEĞER ANALİZİ #################

# ÇOK DEĞİŞKENLİ KAVRAMINI AÇIKLAYACAK OLURSAK : ayrı ayrı baktığımız zaman yaş = 17 olması durumu normal, evlenme durumu= 3
# olması da normalken yaşı 17 olup 3 defa evlenmesi durumu anormal bir durumdur bu yüzden
# ayrı ayrı değerlendirdiğimizde normal olan fakat çok değişkenli bir şekilde değerlendirildiğinde anormallik ortaya çıkarabilir.

# Not
# Elimizde 100 tane değişken var ve ben bunu 2 boyutta görselleştirmek istiyorum bunu nasıl yapabilirim : PCA Yöntemi ile yapılabilir
# PCA (Principal Component Analysis) methodu yüksek boyutlu bir veri setinin boyutunu azaltmak için kullanılan en yaygın yöntemlerden biri.
# *********************************************

df = sns.load_dataset("diamonds")
df = df.select_dtypes(include=['float64','int64'])
df = df.dropna()
df.head()
df.shape
for col in df.columns:
    print(col, check_outlier(df,col))

low, up = outliers_treshold(df,'carat')
df[(df['carat'] > up) | (df['carat']< low)].shape  #1889 tane aykırı değer var


low, up = outliers_treshold(df, "depth")
df[(df['depth'] > up) | (df['depth']< low)].shape

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_
# df_scores = - df_scores  negatif değerler görmek istemezsek
df_scores[0:5]

np.sort(df_scores)[0:5] #-1 e en yakın daha iyi

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked = True, xlim = [0, 50], style = '.-')
plt.show()  #grafiği incelediğimizde eşik değere karar verebiliriz daha keskin geçişi olan kısma dikkat edebiliriz

# eşik değer belirledik
th = np.sort(df_scores)[3] #  3. indexteki değeri seçtik

df[df_scores < th]
df[df_scores < th].shape  # sonuca baktıgımızda 3 tane aykırı gözlem olduğunu görüyoruz daha önce baktığımızda 2500 gibi adette aykırı gözlem vardı

# şimdi bi inceleyelim
df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T
df[df_scores < th].index

# aykırı değerlerden kurtulmak için
df[df_scores < th].drop(axis=0, labels=df[df_scores< th].index) # silebiliriz
#gözlem sayısı çok yüksek olursa baskılama yöntemi karışıklığa sebep olabilir, gözlem sayısı az olursa silme yöntemi uygulanabilir
# Ağaç yöntemi kullanıyorsak aykırı değerlere dokunmamayı tercih ediyoruz ya da dokunacaksak çok ucundan (qr = 99'a qr=1 lik  )


