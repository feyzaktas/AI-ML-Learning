import pandas as pd
import matplotlib.pyplot as plt
dataframe=pd.read_excel("C:\\kod\\YAPAY_ZEKA\\genel_tekrar\\math.excel") #dataset dataframe'e dönüştü
print(dataframe) 
dataframe.info() #dataframe'in billgileri geldi

dataframe.dropna() #boş sütunları sildi
dataframe.drop("Student ID", axis=1) #Student ID bilgisi işimize yaramayacağı için o sütunu sildik.
dataframe.info()

#kategorik verileri sayısayl veriye dönüştürme
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

dataframe["Gender"]=le.fit_transform(dataframe["Gender"])
dataframe["Race/ethnicity"]=le.fit_transform(dataframe["Race/ethnicity"])
dataframe["Parental level of education"]=le.fit_transform(dataframe["Parental level of education"])
dataframe["Lunch"]=le.fit_transform(dataframe["Lunch"])
dataframe["Test preparation course"]=le.fit_transform(dataframe["Test preparation course"])

#----------------------------------------


#standardizasyon(Standard Scaling- Standart Ölçeklendirme)
from sklearn.preprocessing import StandardScaler
import numpy as np
ss=StandardScaler()
scores=["Reading score", "Writing score"]
dataframe[scores]=ss.fit_transform(dataframe[scores])

#feature extraction (class belirleme)
dataframe["Math score"]=pd.cut(dataframe["Math score"], bins=[0,25,50,75,100], labels=[0,1,2,3])
# pd.cut işlevi, bir sayısal sütunu belirli aralıklara bölmek için kullanılır. 
# İlk argüman olarak "Math score" sütunu verilir ve ikinci argüman olarak bins parametresi belirtilir. 
# bins parametresi, aralıkların sınırlarını tanımlar. 
# Örneğin, [0, 25, 50, 75, 100] ifadesi, 0'dan 25'e kadar olan notları bir aralığa, 
# 25'ten 50'ye kadar olan notları bir sonraki aralığa ve böyle devam ederek notları dört farklı aralığa böler.

# Üçüncü argüman olarak labels parametresi belirtilir.
# Bu parametre, her aralık için atanan etiketleri içerir. 
# Yani, 0'dan 25'e kadar olan notlar "0" etiketi ile temsil edilirken, 
# 25'ten 50'ye kadar olan notlar "1" etiketi ile temsil edilir ve benzer şekilde devam eder.

XSample=dataframe.iloc[:,[0,1,2,3,4,6,7]] #özellikleri xsample'a atadık, 5. sütunu class yapacagımız içn almadık
ySample=dataframe["Math score"] #ysample'a class'ı atadık
print("----------------------")
print(XSample)
print(ySample)

#VERİ SETLERİNİ BÖLÜMLENDİRME(TRAİN-VAL.-TEST)
from sklearn.model_selection import train_test_split
XTrain, XTest, yTrain, yTest=train_test_split(XSample,ySample,test_size=0.2,random_state=21)
#test_size: verinin yüzde kaçı test için ayrılacak, random_state: verinin nasıl bölümlendirillecek
print(XSample)
print(XTrain)
print(XTest)

#ALGORTİMALAR(AŞAĞIDA 6 ADET ALGORTİMA ÇEŞİDİ VARDIR)

#1.K-NN(EN YAKIN KOMŞU ALGORİTMASI)
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
  #Train Kısmı
classifier=KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=1)
#n_neighbors: kaç komşuya göre karar verecek, metric: hangi uzaklığı kullanacak, p: hangi parametreyi kullanacak(p=2 öklid, p=1 manhattan)
classifier.fit(XTrain,yTrain) #classifier artık eğitildi, 
# fit: eğitim işlemi, yTrain: gerçek hayattaki sonuçlar
#ylerin hepsi classlarla ilgiliyken xler attiribute
#fit ile train işlemine girer

  #Test Kısmı
yTestPred=classifier.predict(XTest)
#yTestPred benim tahminlediğim sonuçlar, XTest ile hiç görmediği verileri veriyorum.
print(metrics.classification_report(yTest, yTestPred)) #performans değerlendirmeis için karşılaştırma
#yTest: gerçek cevaplar yTestPred: benim tahmniledğim cevaplar

cm=metrics.confusion_matrix(yTest,yTestPred,normalize="true").round(2)
# round: ondalıkta kaç sayı gözükecek
# yTest: Gerçek etiketlerin bulunduğu dizi veya Seri.
# yTestPred: Modelin tahmin ettiği etiketlerin bulunduğu dizi veya Seri.
# normalize="true": Karmaşıklık matrisinin normalize edilmesini sağlar. 
# Bu, her sınıfın gerçek etiketlere göre tahmin doğruluğunu yüzde olarak ifade eder.
disp=metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp=disp.plot(cmap=plt.cm.GnBu, values_format="g") 
#matrix'i görselleştiriyoruz, cmap: renk haritasını belircez.
plt.show()

#2.DECİSİON TREE(KARAR AĞACI ALGORİTMASI)
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

classifier=DecisionTreeClassifier(criterion="log_loss", splitter="best", random_state=42)
#criterion: Bu kriter, bir düğümün bölünmesinin, belirli bir özellikle ilgili veri kümesi içindeki belirsizliği ne kadar azalttığını ölçer
#criterion parametreleri: gini, entropi, log loss
#entropy: 0-1 aralığındadır. entropi ne kadar yüksekse karmaşıklık o kadar fazladır. entropi olabildiğince düşük olmalıdır.
#gini: rastegele etiketlenen verilerin yanlış etiketlenme sıklığını ölçer. 0-0.5 aralığındadır. entropiye göre daha hızlı çalışır. sklearn'e göre default kriterr değeridir.
#log loss: (yeterli bilgi yok, ozan  hocaya sor)
#splitter: bir düğümü nasıl böleceğini belirler. parametreleri: best ve random
#best: düğüm bölünmesinde en iyi özellik seçimini ifade eder. En iyi özellik, belirli bir kriter (örneğin, gini veya entropy) kullanılarak seçilir. 
#genellikle daha iyi sonuçlar verir ancak bölme işlemi daha maliyetlidir.
#random: özellikle büyük veri setleri üzerinde çalışırken ve eğitim süresini azaltmak istediğinizde kullanışlı olabilir.
#bazen "best" seçeneğine göre daha az performanslı olabilir.
classifier.fit(XTrain,yTrain)

#Test Kısmı
yTestPred=classifier.predict(XTest)
print(metrics.classification_report(yTest, yTestPred))
cm=metrics.confusion_matrix(yTest,yTestPred,normalize="true").round(2)
disp=metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp=disp.plot(cmap=plt.cm.GnBu, values_format="g") 
tree.plot_tree(classifier)
plt.show()


#3.RANDOM FOREST ALGORİTMASI
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=101,max_depth=5,random_state=42) 
#n_estimators: kullanılacak olan karar ağaçlarının sayısını belirtir.
#max_depth: her bir karar ağacının maksimum derinliğini belirtir, o sayıya göre dala ayrılır.
classifier.fit(XTrain,yTrain)

#Test Kısmı
yTestPred=classifier.predict(XTest)
print(metrics.classification_report(yTest, yTestPred))
cm=metrics.confusion_matrix(yTest,yTestPred,normalize="true").round(2)
disp=metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp=disp.plot(cmap=plt.cm.GnBu, values_format="g") 
plt.show()

#4.SUPPORT VECTOR MACHİNE(SVM) ALGORTİMASI
from sklearn.svm import SVC
classifier=SVC(c=1, kernel="linear", gamma= 0.1)
#c: margin büyüklüğünü belirler, c arttıkça margin daralır.
#kernel: elimizde karmaşık datasetler varsa kernel ile ayırırız. kerneller: linear, rbf, polynomial kernel, sigmoid
#gamma:çekirdek genişliğini kontrol eder ve aşırı uyum (overfitting) ya da aşırı basitleştirme (underfitting) arasında bir denge kurar.
#düşük değerleri, geniş bir çekirdek anlamına gelirken, yüksek değerler daha dar bir çekirdek anlamına gelir.
classifier.fit(XTrain,yTrain)

#Test Kısmı
yTestPred=classifier.predict(XTest)
print(metrics.classification_report(yTest, yTestPred))
cm=metrics.confusion_matrix(yTest,yTestPred,normalize="true").round(2)
disp=metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp=disp.plot(cmap=plt.cm.GnBu, values_format="g") 
plt.show()

#5.NAVİE BAYES ALGORİTMASI
from sklearn.naive_bayes import BernoulliNB
classifier=BernoulliNB()
#naviebayes algoritması basit bir algoritma old. için parametreeye gerek yoktur.
classifier.fit(XTrain,yTrain)
yTrainPred=classifier.predict(XTrain)

#Test Kısmı
yTestPred=classifier.predict(XTest)
print(metrics.classification_report(yTest,yTestPred))
cm=metrics.confusion_matrix(yTest,yTestPred,normalize="true").round(2) 
disp=metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp=disp.plot(cmap=plt.cm.GnBu,values_format="g") 
plt.show()

#6.LOJİSTİK ALGORİTMA
from sklearn.linear_model import LogisticRegression
from sklearn import tree
classifier=LogisticRegression(random_state=0, max_iter=1000,solver="newton-cg")
classifier.fit(XTrain,yTrain)

#Test Kısmı
yTestPred=classifier.predict(XTest) 
print(metrics.classification_report(yTest,yTestPred))
cm=metrics.confusion_matrix(yTest,yTestPred,normalize="true").round(2)
disp=metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp=disp.plot(cmap=plt.cm.GnBu,values_format="g") 
plt.show()