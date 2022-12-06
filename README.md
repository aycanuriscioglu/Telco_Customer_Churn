Business Problem [EN]

It is desirable to develop a machine learning model that can predict customers who will leave the company. Before developing the model, you are expected to perform the necessary data analysis and feature engineering steps.
İş Problemi [TR]

Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir.  Modeli geliştirmeden öncegerekli olan veri analizi ve özellik mühendisliği adımlarını gerçekleştirmeniz beklenmektedir.


#######################################################################################
Data Set Story (Veri Seti Hikayesi)

[EN]

Telco customer loss data includes information about a fictitious telecom company that provided home phone and Internet services to 7043 customers in California in the third quarter. It shows which customers have left their service, stayed, or signed up for the service
Veri Seti Hikayesi

[TR]

Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve İnternet hizmetleri sağlayan hayali bir telekom şirketi hakkında bilgi içerir. Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını ve yahizmete kaydolduğunu gösterir

###############

Veri Seti Hikayesi

CustomerId: Müşteri İd’si

Gender: Cinsiyet

SeniorCitizen: Müşterinin yaşlı olup olmadığı (1, 0)

Partner: Müşterinin bir ortağı olup olmadığı (Evet, Hayır)

Dependents: Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır

tenure: Müşterinin şirkette kaldığı ay sayısı

PhoneService: Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)

MultipleLines: Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)

InternetService: Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)

OnlineSecurity: Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)

OnlineBackup: Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)

DeviceProtection: Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)

TechSupport: Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)

StreamingTV: Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok)

StreamingMovies: Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok)

Contract: Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)

PaperlessBilling: Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)

PaymentMethod: Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))

MonthlyCharges: Müşteriden aylık olarak tahsil edilen tutar

TotalCharges: Müşteriden tahsil edilen toplam tutar

Churn: Müşterinin kullanıp kullanmadığı (Evet veya Hayır
