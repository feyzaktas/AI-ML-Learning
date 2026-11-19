import matplotlib.pyplot as plt
import numpy as np
from random import randint as rnd

plt.title("Aylara Göre İhracat-İthalat")
plt.xlabel("Aylar")
plt.ylabel("İhr.-İth.")
plt.grid()

xpoint=np.array(["Ocak","Subat","Mart","Nisan","Mayıs","Haziran","Temmuz","Ağustos","Eylül", "Ekim","Kasım","Aralık"])
ypointIhracat=np.array([rnd(500,1000) for i in xpoint])
ypointIthalat=np.array([rnd(700,1200) for i in xpoint])

plt.plot(ypointIhracat, linestyle="--", color="y",lw="3",marker="o", ms=5, mec="r", mfc="r")
plt.plot(ypointIthalat , linestyle="-", color="c",lw="5",marker="d", ms=7, mec="b", mfc="b")
plt.show()



