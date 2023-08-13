########## EKSİK DEĞERLER ##########
# Silme # Değer Atama Yöntemleri #Tahmine Dayalı Yöntemler

# NOT : Eksik değerlere sahip gözlemlerin veri setinden direkt çıkarılması ve
# rassallığının incelenmemesi, yapılacak istatistiksel çıkarımların ve
# modelleme çalışmalarının güvenirliğini düşürecektir.

# NOT : Eksik gözlemlerin veri setinden direkt çıkarılabilmesi için
# veri setindeki eksikliğin bazı durumlarda kısmen bazı durumlarda
# tamamen  raslantısal olarak oluşmuş olması gerekmektedir.

# NOT: Eğer eksiklikler değişkenler ile ilişkili olarak ortaya çıkan
# yapısal problemler ile meydana gelmiş ise bu durumda yapılacak silme
# işlemleri ciddi yanlılıklara sebep olabilecektir.


################# EKSİK DEĞERLERİ YAKALAMA ####################

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
def grab_col_names(dataframe, cat_th = 10, car_th = 20):
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

# eksik gözlem var mı yok mu sorgusu
df.isnull().values.any()

# değişkenlerdeki eksik değer sayisi
df.isnull().sum()

# en az bir tane eksik değere sahip olan gözlem birimleri

df[df.isnull().any(axis=1)]

df[df.notnull().all(axis=1)] # tüm sütunlarında NA değeri olmayan satırları getir

#Azalan şekilde sıralamak istersek
df.isnull().sum().sort_values(ascending=False)

df.shape[0] # satır
df.shape[1] # sütun

# eksik değerleri oran olarak göstermek istersek
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending= False)

# eksik değerleri içeren columnları alalım
na_columns = [col for col in df.columns if df[col].isnull().sum() > 0]


#tüm bu yaptıklarımızı kolaylaştırmak adına fonksiyonel bir şekilde yazalım
def missing_value_table(dataframe, na_name = False):
    na_column = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_column].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending= False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss','ratio'])
    print(missing_df, end='\n')

    if na_name:
        return na_column

missing_value_table(df)
missing_value_table(df, True)

#################### EKSİK DEĞER PROBLEMİNİ ÇÖZME #####################

# ÇÖZÜM 1 : HIZLICA SİLMEK

df.shape
df.dropna().shape

# ÇÖZÜM 2 : BASİT ATAMA YÖNTEMLERİ İLE DOLDURMAK

df["Age"].fillna(df["Age"].mean())
df["Age"].fillna(df["Age"].mean()).isnull().sum() # 0

# dataframe deki NA değerlerine sahip değişkenlerin hepsine tek bir fonksiyonla uygulamak istersek

# df.apply(lambda x: x.fillna(x.mean()), axis=0) bu şekilde yaparsak hata alırız çünkü veri setimizin içerisinde object değerler de var

df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0).head()
dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
dff.isnull().sum().sort_values(ascending=False)
# Sayısal değişken olan Age için mean kullanarak eksik değer problemini çözebildik
# fakat kategorik değişken olan cabin ve embarked değişkenlerinin hala eksik değerleri duruyor

# kategorik değişkenin mode ' unu alabiliriz

df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()
df["Embarked"].fillna("missing") # şeklinde de eksikleri doldurabiliriz

dfff = df.apply(lambda x: x.fillna(x.mode()[0]) if x.dtype == "O" and len(x.unique()) <= 10 else x, axis=0)# kategorik değişken olup olmadıgını anlamak için < 10 dedik çünkü kardinal değişken olma durumu da var

dfff.isnull().sum().sort_values(ascending=False)  # embarked değişkenimizin eksik değerleri doldurulmuş oldu
dfff
##################  KATEGORİK DEĞİŞKEN KIRILIMINDA DEĞER ATAMA ###################
# burada yapmak istediğimizi bir örnek üzerinden geçecek olursak eksik değerleri cinsiyetin ortalamasıyla doldurmak yerine
# cinsiyeti kadın ve erkek olan gruplarına kırarak ayrı ayrı ortalamalarına bakılarak ayrı ayrı eksik değerleri doldurabiliriz



df.groupby("Sex"["Age"]).mean()
df["Age"].mean()

df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()  # transform("mean") ortalaması ile değiştir

# Daha açık şekliyle yapmak istersek
df.groupby("Sex")["Age"].mean()["female"]  # kadınlara göre yaş ortalaması

df.loc[(df["Age"].isnull()) & (df["Sex"] == "female")] #cinsiyeti kadın olup NaN olan değerleri getir

df.loc[(df["Age"].isnull()) & (df["Sex"] == "female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"] # kadınlardan NaN olan değerleri kadınların yaş ortalaması değerleriyle doldurduk

df.loc[(df["Age"].isnull()) & (df["Sex"] == "male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"] # erkeklerden NaN olan değerleri erkeklerinyaş ortalaması değerleriyle doldurduk

df.isnull().sum()

# Çözüm 3
#################### TAHMİNE DAYALI ATAMA İŞLEMİ #####################

# makine modelleme meetodları kullanılacaktır

df = load()

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True) #Burada kategorik değişkenleri numeric şekilde ifade ediyoruz

dff.head()

# Değişkenlerin standartlaştırılması
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()

# KNN'in uygulanması
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5) #en yakın 5 komşusuna bakıp eksik değerleri ona göre dolduracak
dff = pd.DataFrame(imputer.fit_transform(dff), columns =dff.columns) # eksiklikleri doldurduk
dff.head()

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns) # burada scaler yapmadan önceki haline geri çevirdik


df["age_imputed_knn"] = dff[["Age"]]

df.head()

df.loc[df["Age"].isnull(), ["Age","age_imputed_knn"]]


#### RECAP ######

df = load() # veri setini yükledik
missing_value_table(df) # eksik verileri raporladık
# sayısal değişkenleri direkt medyan ile doldurduk
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()
# kategorik değişkenleri mode ile doldurduk
df.apply(lambda x: x.fillna(x.mode()[0]) if x.dtype == "O" and len(x.unique() <= 10) else x, axis=0).isnull().sum()
# kategorik değişken kırılımında sayısal değişkenleri doldurduk
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# tahmine dayalı atama ile Doldurma

############# EKSİK VERİNİN YAPISI İNCELEMEK #################

# Gelişmiş Analizler

msno.bar(df)
plt.show()  # eksik değerlerin olduğu değişkenleri gözlemleyebiliyoruz / veri setindeki gözlemlerin sayılarını gözlemleyebiliyoruz

msno.matrix(df) # değişkenlerdeki eksikliklerin birlikte çıkıp çıkmadıdğını gösterir
plt.show()

msno.heatmap(df) # değişkenlerdeki eksik değerlerin birbiriyle ilişkisine (korelasyonu) bakabiiliriz
plt.show()

######### EKSİK DEĞERLERİN BAĞIMLI DEĞİŞKEN İLE ANALİZİ ##########

missing_value_table(df,True)
na_cols = missing_value_table(df,True)


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + 'NA_FLAG'] = np.where(temp_df[col].isnull(),1,0) # eksiklik gördüğü yere 1 görmediği yere 0 yazar
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("NA")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN" : temp_df.groupby(col)[target].mean(),
                            "COUNT": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df,"Survived",na_cols)
# Cabin değişkeninde normalde %70 gini büyük bir oranda eksik değer vardı ve bu değişkeni dataframe'mimizden çıkarmak isteyebilirdik
# sonuçlara baktığımızda  Cabin değişkeninde eksik olmayan değerlerin hayatta kalma yüzdesi yükse buradan çıkaracağımı Sonuç şu ki
# eğer bu şekilde veriyi incelemeseydik direkt Cabin değişkenini silebilirdik ama sonuçlara bakılınca NaN olmayan değerler bizim için
# önemli bilgiler taşıyor bu yüzden silmek bizim için bilgi kaybına neden olacaktı

