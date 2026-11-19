import numpy as np

x1=np.array([0,0,1,1])
x2=np.array([0,1,0,1])
y1=np.array([0,1,1,1])
w1=2
w2=3
trash=1

sonuc=np.array(["","",""])

for i in range(0,4):
    toplam=(x1[i]*w1)+(x2[i]*w2)
    if toplam < trash:
        sonuc[i]=0
    else:
        sonuc[i]=1

print(y1)
print(sonuc)

#herbbian öğrenme
import numpy as np

# Örnek veri kümesi (örneğin iki nöronun çıktılarını temsil eder)
veri = np.array([
    [1, -1],  # İlk örnek, iki nöron da aktif
    [-1, 1],  # İkinci örnek, yalnızca ikinci nöron aktif
    [1, 1],   # Üçüncü örnek, her iki nöron da aktif
])

# Sinaptik bağlantıları rastgele başlatın
sinaptik_baglanti = np.random.random((2, 2))
                                     

# Öğrenme kuralını uygula
for veri_ornek in veri:
    # Her bir örnek için bağlantıları güncelle
    sinaptik_baglanti += np.outer(veri_ornek, veri_ornek)

print("Son sinaptik bağlantı matrisi:")
print(sinaptik_baglanti)