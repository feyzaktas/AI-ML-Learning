import pandas as pd
from random import randint as rd

data={"Sabah": [rd(800,1400)for i in range(20)],"Ogle": [rd(800,1400)for i in range(20)],"Akşam": [rd(800,1400)for i in range(20)],}
frame=pd.DataFrame(data, index=["Gün"+ str(i) for i in range(20)])

print(frame)
#frame.info()
#print(frame.head()) #ilk 5
#print(frame.tail()) #son 
#print(frame.sample(frac=0.5)) 

print(frame["Sabah"]) #sabah sütununu yazdırır.
print(frame["Sabah"][5]) #sabah sütununun 5. indexteki elemanını yazdırır.
print(frame["Sabah"][5:15]) #sabah sütunundaki 5 ve 15. index aralığını yazdırır.
print(frame[["Sabah","Ogle"]][5:15]) #ögle ve sabah sütunlarının 5 ve 15. index aralıgını yazdırrır.
print(frame.loc["Gün5"]) #satır bazında indexliyoruz. tüm satırı alıyoruz
print(frame.iloc[0:15]) #index numarası üzerinden yürür. 
#loc ve iloc satırları indexler

print(frame.iloc[[2,4,6,9]]) #sayılarla da cekebiliyoruz

#frame[ozellik].[ornek]
#fram.loc[index ismi]
#frame.iloc[index no]

#THE END OF PANDAS

#MATPLOTLİB

import matplotlib.pyplot as plt
import numpy as np
from random import randint as rnd

plt.xlabel("Yıllar") #x ekseninin isimini belirledik
plt.ylabel("İhracaat") #y ekseninin ismini belirledik
plt.title("Yıllara Göre İhracat Rakamları") #grafigin baslıgını belirledik

xpoint=np.array([i for i in range (2010, 2016)]) #x ekseninin aralıgı 2010 ve 2016 olarak belirlendi
ypoint=np.array([rnd(800, 12000) for i in range(6)]) #y ekseninin aralıgı 800 ve 1200 arasındaki rastgele 6 sayı olarak belirlendi

plt.plot(xpoint, ypoint, "-.", marker="o",ms=10, mfc="r") #grafigi görsellestirdik
#çizginin stilini (noktalı, kesik vb.) " " içinde yazarak gösteriririz.
#color: cizginin rengini belirliyoruz
#linewidth=çizginin kalınlıgı, integer girilir (lw)
#marker= grafikteki belirtilecek yerdeki simgeyi ayarlar.
#ms=simgenin büyüklügü, integer girilir (marker size)
#mec=simgenin dış rrengi
#mfc=simgenin iç rengi

plt.show()