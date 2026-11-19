import numpy as np
import matplotlib.pyplot as plt
from random import randint as rnd
font1={"family":"arial", "color":"red", "size":18} #tema tanımlama d,ct. formatında
font2={"family":"serif","color":"darkblue", "size":15}

plt.title("Accuracy Graph", fontdict=font1, loc="center") 
plt.xlabel("Epoch",fontdict=font2,loc="left")
plt.ylabel("Accurary", fontdict=font2, loc="bottom")
plt.grid()
yTrainPoint=([0.5,0.65,0.77,0.85,0.92,0.97,0.98,0.99])
yValPoint=([0.35,0.55,0.70,0.75,0.80,0.85,0.85,0.88])

plt.plot(yTrainPoint,linestyle=":",color="r",lw="2",marker="x")
plt.plot(yValPoint,linestyle="-",color="y",lw="4",marker="o")

plt.legend(["Train Accuracy", "Val Accuracy"],loc=0)
plt.show()

# #COKLU GRAFİK YAPMA
# yTrainPoint1=np.array([0.8,0.7,0.55,0.50,0.45])
# plt.subplot(1,3,1) #1
# plt.plot(yTrainPoint1)
# plt.title("Loss Graph")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")


# yTrainPoint2=np.array([0.6,0.5,0.32,0.12,0.01])
# plt.subplot(1,3,2) # (bir satır, 3 sütun, 2.grafik)
# plt.plot(yTrainPoint1)
# plt.title("Loss Graph")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")


# yTrainPoint3=np.array([1.5,1.2,0.88,0.75,0.50])
# plt.subplot(1,3,3) 
# plt.plot(yTrainPoint1)
# plt.title("Loss Graph")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")

# plt.show()

#DİK SÜTUN GRAFİĞİ 
# plt.title("Yıllık Meyve Tüketimi")
# x1 = np.array(["Elma","Armut", "Muz","Şeftali","Ananas"])
# y1= np.array([120,30,110,60,10])

# plt.bar(x1,y1,color=["red", "yellow", "green", "blue", "pink"],
#         width=0.5) #bar=dik sütun barh=yatay, width=uzunluk height=genislik
# plt.show()

#HISTOGRAM
y2=np.array([rnd(1,100) for i in range (10000)])
x2=np.array([i for i in range(1,101)])
plt.title("Histogram")
print(y2)

plt.hist(y2,x2, color="red") 
plt.show()

#PASTA GRAFİĞİ
pie=np.array([25,30,15,5,90])

label=np.array(["Toyata","Hyundai","Fiat","Opel", "Honda"])

plt.pie(pie,labels=label, colors=["darkred", "yellow", "green", "orange",
                                  "darkblue"],explode=[0.2,0,0,0,0],
                                  shadow=True, autopct="%1.1f%%") #yüzdelik kısmı sayı görünümünü temsil eder, shadow:gölge kısmı, 

plt.legend(title="--Arabalar")
plt.show()






