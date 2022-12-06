########################################## Telco Churn Feature Engineering ##########################################

#################################################
# İş Problemi
#################################################

# Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli
# geliştirilmesi istenmektedir. Modeli geliştirmeden önce gerekli olan veri analizi
# ve özellik mühendisliği adımlarını gerçekleştirmeniz beklenmektedir.

#################################################
# Veri Seti Hikayesi
#################################################

# Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu
# ve İnternet hizmetleri sağlayan hayali bir telekom şirketi hakkında bilgi içerir. Hangi
# müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu
# gösterir.

#################################################
# Değişkenler
#################################################

# CustomerId: Müşteri İd’si
# Gender: Cinsiyet
# SeniorCitizen: Müşterinin yaşlı olup olmadığı (1, 0)
# Partner: Müşterinin bir ortağı olup olmadığı (Evet, Hayır)
# Dependents: Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır
# tenure: Müşterinin şirkette kaldığı ay sayısı
# PhoneService: Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
# MultipleLines: Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
# InternetService: Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
# OnlineSecurity: Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# OnlineBackup: Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# DeviceProtection: Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# TechSupport: Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingTV: Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingMovies: Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# Contract: Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
# PaperlessBilling: Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
# PaymentMethod: Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
# MonthlyCharges: Müşteriden aylık olarak tahsil edilen tutar
# TotalCharges: Müşteriden tahsil edilen toplam tutar
# Churn: Müşterinin kullanıp kullanmadığı (Evet veya Hayır

############################################ PROJE GÖREVLERİ ############################################

#################################################
# GÖREV 1 : Keşifçi Veri Analizi
#################################################

# Adım 1: Genel resmi inceleyiniz.
# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
# Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.
# Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre
# numerik değişkenlerin ortalaması)
# Adım 5: Aykırı gözlem analizi yapınız.
# Adım 6: Eksik gözlem analizi yapınız.
# Adım 7: Korelasyon analizi yapınız.

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
warnings.simplefilter(action="ignore")
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

def load_application_train():
    data = pd.read_csv("feature_engineering/Telco-Customer-Churn.csv")
    return data

df = load_application_train()



def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

for column in df.columns:
    print(f"Column: {column} --> Unique Values: {df[column].nunique()}")

df[df["tenure"]==0]

# Object türdeki TotalCharges değişkenini numerik olarak değiştirmeliyiz. Default olarak, bu işlev sayısal olmayan
# verileri gördüğünde bir istisna oluşturur; ancak, bu durumları atlamak ve onları bir
# NaN ile değiştirmek için error='coerce' argümanını kullanabiliriz.

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors='coerce')

df.info()

df[df["TotalCharges"].isnull()]

# customerID değişkeni müşterinin churn olup olmayacağını açıklamaz...
df.drop(columns="customerID", inplace=True)

df["Churn"].unique()
df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

# Numerik ve Kategorik Değişkenlerin Yakalanması

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Kategorik Değişken Analizi

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col, plot=True)

# Numerik Değişken Analizi

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)

# Tenure değişkeni incelendiğinde en çok 1 aylık müşteri sayısının fazla olduğunu ve ikinci sırada da 72 aylık
# müşteri sayısının geldiğini görüyoruz.

df["tenure"].value_counts().head()

# Bu durum, Contract' tan (Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)) kaynaklanıyor olabilir.
# Aydan aya sözleşme yenileyen müşterilerin tenure değerlerine bakalım:
df[df["Contract"] == "Month-to-month"]["tenure"].hist(bins=20)
plt.xlabel("tenure")
plt.title("Month-to-month")
plt.show()

# Sözleşmesi 2 yıllık olan müşterilerin tenure değerlerine bakalım:
df[df["Contract"] == "Two year"]["tenure"].hist(bins=20)
plt.xlabel("tenure")
plt.title("Two year")
plt.show()

# Aydan aya sözleşme yenileyen müşterilerin MonthyChargers değerlerine bakalım:
df[df["Contract"] == "Month-to-month"]["MonthlyCharges"].hist(bins=20)
plt.xlabel("MonthlyCharges")
plt.title("Month-to-month")
plt.show()

df[df["Contract"] == "Month-to-month"]["MonthlyCharges"].mean() # 66.39849032258037

# Sözleşmesi 2 yıllık olan müşterilerin MonthyChargers değerlerine bakalım:
df[df["Contract"] == "Two year"]["MonthlyCharges"].hist(bins=20)
plt.xlabel("MonthlyCharges")
plt.title("Two year")
plt.show()

df[df["Contract"] == "Two year"]["MonthlyCharges"].mean() # 60.770412979351
# Aydan aya sözleşme yenileyen müşterilerin aylık ortalama ödemeleri, 2 yıllık sözleşme yaptıranlardan daha fazla.

# Kategorik Değişkenlerin Target Değişken ile Analizi

def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)

# Nümerik Değişkenlerin Target Değişken ile Analizi

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Churn", col)

# Aykırı Değer Analizi

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

for col in num_cols:
    print(col, check_outlier(df, col))

# Eksik Gözlem Analizi

df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)

df[df["TotalCharges"].isnull()]

df.dropna(inplace=True)

df.isnull().sum()

df.shape

# Korelasyon Analizi

f, ax = plt.subplots(figsize=[14, 8])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f",cmap="coolwarm")
ax.set_title("Correlation Matrix", fontsize=25)
plt.show()

#################################################
# GÖREV 2: FEATURE ENGINEERING
#################################################

# Adım 1:  Eksik ve aykırı değerler için gerekli işlemleri yapınız.
# Adım 2:  Yeni değişkenler oluşturunuz.
# Adım 3:  Encoding işlemlerini gerçekleştiriniz.
# Adım 4:  Numerik değişkenler için standartlaştırma yapınız.
# Adım 5:  Model oluşturunuz.

# müşterilerin tenure değerlerine göre yıllık alt kategorilerden oluşan yeni bir değişken oluşturma
df.loc[(df["tenure"] >= 0) & (df["tenure"] <= 12), "TENURE_YEAR_FLAG"] = "0-1 Year"
df.loc[(df["tenure"] > 12) & (df["tenure"] <= 24), "TENURE_YEAR_FLAG"] = "1-2 Year"
df.loc[(df["tenure"] > 24) & (df["tenure"] <= 36), "TENURE_YEAR_FLAG"] = "2-3 Year"
df.loc[(df["tenure"] > 36) & (df["tenure"] <= 48), "TENURE_YEAR_FLAG"] = "3-4 Year"
df.loc[(df["tenure"] > 48) & (df["tenure"] <= 60), "TENURE_YEAR_FLAG"] = "4-5 Year"
df.loc[(df["tenure"] > 60) & (df["tenure"] <= 72), "TENURE_YEAR_FLAG"] = "5-6 Year"
df.groupby("TENURE_YEAR_FLAG")["Churn"].mean()
df.head()

# Müşterilerin otomatik ödeme yapıp yapmadıkları bilgisini veren değişken tanımı
df["AUTOMATIC_PAYMENT_FLAG"] = df["PaymentMethod"].apply(lambda x: 1 if x in ["Credit card (automatic)", "Bank transfer (automatic)"] else 0)
df.groupby("AUTOMATIC_PAYMENT_FLAG")["Churn"].mean()
df.head()

# Bakmakla yükümlü olduğu kişiler var, çoklu hat kullanmıyor
# df["YESDEPENDENTS_NOMULTIPLE_FLAG"] = df.apply(lambda x: 1 if (x["Dependents"] == "Yes") and (x["MultipleLines"] != "Yes") else 0, axis=1)
# df.groupby("YESDEPENDENTS_NOMULTIPLE_FLAG")["Churn"].mean()

# Aydan aya sözleşme yenileyen genç müşteriler
df["YOUNG_MONTH_TO_MONTH_FLAG"] = df.apply(lambda x: 1 if (x["SeniorCitizen"] == 0) and (x["Contract"] == "Month-to-month") else 0, axis=1)
df.groupby("YOUNG_MONTH_TO_MONTH_FLAG")["Churn"].mean()

# df.columns = [col.upper() for col in df.columns]

# Encoding

cat_cols, num_cols, cat_but_car = grab_col_names(df)

#############################################
# Label Encoding
#############################################

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

#############################################
# Rare Encoding
#############################################

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "Churn", cat_cols)

#############################################
# One-Hot Encoding
#############################################

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df,ohe_cols,drop_first=True)

df.head()
df.info()

# Nümerik Değişkenler İçin Standartlaştırma Yapılması

# Standartlaştırma, nümerik sütunları ortak bir ölçeğe dönüştürmekten oluşan makine öğreniminde
# yaygın kullanılan bir uygulamadır.

# StandardScaler: Klasik standartlaştırma(normalleştirme).
# Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s (z standartlaştırılması)
# x: bütün değerler, u: ortalama, s: standart sapma, z: standart değer


scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df[num_cols].head()


# Model Oluşturma

y = df["Churn"] # bağımlı değişken
X = df.drop(["Churn"], axis=1) # bağımsız değişkenler
# veri setini train ve test olarak ikiye ayırıyoruz.
# train seti üzerinde model kuracağız, test seti ile kurduğumuz modeli test edeceğiz.
# model bütün veriyi görerek ezber yapmasın, bir setiyle eğitilsin, diğer setiyle test edilsin.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
# test_size parametresi, test seti için verilerin hangi bölümünün tutulacağına karar verir

from sklearn.ensemble import RandomForestClassifier # ağaç temelli yöntem kullanıyoruz

# Random Forest, regresyon ve sınıflandırma problemlerini çözmek için kullanılan bir makine öğrenimi tekniğidir.

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train) # modeli kurduk
y_pred = rf_model.predict(X_test) # test setindeki X bağımsız değişken değerlerini modele sorduk ve
# Modelden test setinin bağımlı değişkenlerini tahmin etmesini istiyoruz.
accuracy_score(y_pred, y_test) # test setinin y bağımlı değişkeni ile tahmin edilen değerleri kıyaslıyoruz.
# müşterilerin 0.78'i nin Churn olup olamayacağını doğru tahmin eder...
# başarı oranı  %78...

# yeni değişkenlerimiz anlamlı mı anlamsız mı?

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)

# Modelimizin TotalCharges ve tenure'e diğerlerine göre daha fazla önem verdiğini görebiliriz.



