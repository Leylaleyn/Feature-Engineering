# Ham veriden değişken türetmek

# Mesela 2023-02-05 07:45 değişkeninden yıl,ay,gün,saat hatta haftanın hangi günleri olduğu bilgisini çıkartabiliriz

##################### BINARY FEATURES ####################
import pandas as pd
import numpy as np
from Fonk import load
df = load()
df.head()

# veri setimizi incelediğimizde Cabin değişkenimize NaN olanlara 0 NaN olmayan değerlere 1 verelim ve inceleyelim

df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int')

df.groupby("NEW_CABIN_BOOL").agg({"Survived" : "mean"})  # sonuç olarak Cabin değeri 1 olanların Cabin değeri olmayanlara göre hayatta kalma olasılığı daha yüksekmiş
# Daha öncesinde bize çöp gibi gelen Cabin değişkeni aslında önemliymiş !!!

from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count = [df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
                                               df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],
                                      nobs = [df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                              df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])


print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p1 ve p2 arasında fark yoktur hipotezi p-value < 0.05 den küçük olduğundan dolayı reddedilir yani aralarında istatistik olarak anlamlı bir farklılık vardır
#  p1 ve p2 oranları arasında fark vardır bu oranlar : 0  0.300    1  0.667 oranlarıydı

# Çok değişkenli etkisini bilmiyoruz ama şimdiye kadar ki gözlemimize dayanarak bu değişkenle devam edebiliriz gibi gözüküyor

# şimdi başka bir binary feature ler oluşturalım
#SibSp Parch ( bunlar akrabalığı ifade ediyor ) değişkenlerini inceleyelim

df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"

df.groupby("NEW_IS_ALONE").agg({"Survived" : "mean"}) # ailesi olanın hayatta kalma olasılığı daha yüksek gibi duruyor


test_stat, pvalue = proportions_ztest(count = [df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                               df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],
                                      nobs = [df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                              df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])


print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# bu değişkende de p-value < 0.05 imiş yani değişkeni göz ardı etmesek iyi olur


################################# TEXT FEATURE #############################

df = load()
df.head()
# Name değişkeni için bakabiliriz

####### Letter Count

df["NEW_NAME_COUNT"] = df["Name"].str.len()  # str.len() ile Serisi içindeki her bir dizenin uzunluğunu hesaplayabiliriz.

###### Word Count

df["NEW_WORD_COUNT"] = df["Name"].apply(lambda x : len(str(x).split(" "))) # str stringe çeviriyor

####### Özel Yapıları Yakalamak
# Doktor olanlar var mı diye bakalım
df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
df.groupby("NEW_NAME_DR").agg({"Survived" : ["mean","count"]}) # doktor olanların hayatta kalma oranları daha yüksekmiş (ama 10 doktor varmış)


############################ REGEX FEATURES ##############################
from datetime import date
df = load()
df.head()

df['NEW_TITLE'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand = False)   # ' ([A-Za-z]+)\.' -> başta boşluk olan büyük ya da küçük harflerden oluşan sonunda nokta ile biten

df[["NEW_TITLE","Survived","Age"]].groupby(["NEW_TITLE"]).agg({"Survived" : "mean",
                                                               "Age" : ["count","mean"]})
######################## DATE FEATURES ##################
##### Date Değişkenleri Üretmek

dff = pd.read_csv("course_reviews.csv")
dff.head()
dff.info()

# Timestamp değişkeni üzerinden üreteceğiz baktığımız zaman object tipinde
# bunu datetime tipine çevirelim

dff["Timestamp"] = pd.to_datetime(dff["Timestamp"], format="%Y-%m-%d %H:%M:%S")  # Yıl Ay Gün olarak

# yıl
df["Year"] = dff["Timestamp"].dt.year

# ay
dff["Month"] = dff["Timestamp"].dt.month

#yıl farkı
dff['Year_Diff'] = date.today().year - dff["Timestamp"].dt.year

# month diff ( iki tarih arasındaki ay farkı) : yıl farkı + ay farkı
dff['Month_Diff'] = (date.today().year - dff["Timestamp"].dt.year) * 12 + date.today().month - dff["Timestamp"].dt.month

# day name
dff["Day_Name"]= dff["Timestamp"].dt.day_name()
dff.head()

######################## ÖZELLIK ETKİLEŞİMLER ( FEATURE INTERACTION) ########################

df = load()
df.head()


df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]  # yaş değişkeni ile yolculuk yapıcağı sınıf değişkenleriyle yeni bir sınıf oluşturdurk

df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1 # akrabalar + 1(kendisi) yeni bir aile değişkeni oluşturduk

# yaşı 21 den küçük erkek için yeni bir değişken oluturabildik
df.loc[(df["Sex"] == "male") & (df["Age"] <= 21), "NEW_SEX_CAT"] = "youngmale"
df.loc[(df["Sex"] == "male") & (df["Age"] > 21) & (df["Age"] <=50), "NEW_SEX_CAT"] = "maturemale"
df.loc[(df["Sex"] == "male") & (df["Age"] > 50), "NEW_SEX_CAT"] = "seniormale"

df.loc[(df["Sex"] == "female") & (df["Age"] <= 21), "NEW_SEX_CAT"] = "youngfemale"
df.loc[(df["Sex"] == "female") & (df["Age"] > 21) & (df["Age"] <=50), "NEW_SEX_CAT"] = "maturefemale"
df.loc[(df["Sex"] == "female") & (df["Age"] > 50), "NEW_SEX_CAT"] = "seniorfemale"

df.groupby("NEW_SEX_CAT")["Survived"].mean()









