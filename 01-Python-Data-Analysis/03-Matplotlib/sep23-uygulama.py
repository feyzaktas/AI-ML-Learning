import matplotlib.pyplot as plt
from random import randint as rnd
import numpy as np

x=np.array([rnd(1,100) for i in range(100)])
y=np.array([rnd(1,100) for i in range(100)])

colors=np.array([rnd(1,100) for i in range(100)])
sizes=10*np.array([rnd(1,100) for i in range(100)])
plt.scatter(x,y, c=colors, s=sizes, alpha=0.5,   #alpha: şefaflık saglar noktalarada 
            cmap='viridis')
plt.colorbar() #bir grafik üzerinde renk sakalasını gözükmesini saglar

plt.show()