import numpy as np
#from numpy import min,max,sum

array=np.array([1,2,3]) #bir numpay array tanımladık
print(array)
print(type(array)) #arraytipi-ndrray
print(array.ndim) #kac boyutlu oldugunu gosterir(1 boyutlu bu array)
print(array.shape) #kac satır kac sütun var onu gösterir (cıktı: (3,))
array2=np.array([[1,2,3],
                [4,5,6]])
print(array2.shape[0]*array2.shape[1]) #iki bpyutlu

zeroArray=np.zeros(5) #5 tane 0
oneArray=np.ones(10) #10 tane 1
fullArray=np.full(15, "hello") 

print(zeroArray)
print(oneArray)
print(fullArray)

print("----------------------------------------")

array3=np.array([[1,2,3], #3 boyutlu
                [4,5,6], 
                [7,8,9]])

zeroArray2=np.zeros((3,3)) #satıra da sutuna da 0 yazdı 3'er tane birinci kısım satır, ikinci kısım sütun
oneArray2=np.ones((3,3)) #1 yazdı
fullArray2=np.full((3,3),12) #12 yazdı

print(zeroArray2)
print(oneArray2)
print(fullArray2)

print("3 BOYUTLU")

array3=np.array([[[1,2,3],
                  [4,5,6]],
                  [[7,8,9],
                  [10,11,12],]]) # 2 tane matrisi birlestirerek 3 boyutlu sekil olusturduk

print(array3)
print(array3.shape) # boyutlatın uzunlugunu aldık. (k,i,j) cıktısı (2,2,3) / matris sayısı aarttıkça  k da artar
print(array3.ndim) #kac boyutlu

zeroArray3=np.zeros((3,2,3))
oneArray3=np.ones((3,2,3))
fullArray3=np.full((3,2,3),"Feyza")

print(zeroArray3)
print(oneArray3)
print(fullArray3)

#İNDEXLEME
#0dan baslıyor

array_2=np.array([[1,2,3],
                  [4,5,6]])

print(array_2[1, :]) # 1. satırdaki tüm elemenalaarı alın
print("---")
print(array_2[ :, 2])# 2. sutundaki elemanlar
print(array_2[0,1:3]) # 0. satırın 1. ve 2. elemanı 

array_3=np.array([[[1,2,3],
                     [4,5,6]],
                     
                     [[7,8,9],
                      [10,11,12]],
                      [[13,14,15],
                       [16,17,18]]])

print(array_3[1,0,:]) # 1.matris 0. satırdaki tüm elemanlar
print(array_3[:,1,0]) # her matrisin 1. satırının 0.elemanı



#VERİ TİPLERİ
array4=np.array([1,2,3,4,5], dtype="i")
print(array4)
print(array4.dtype)

newArray=array4.astype("f") #veritipini degistirir
print(newArray)
print(newArray.dtype)

#REFERANS GOSTERME
oArray=np.array([1,2,3,4,5,6,7,8])
cArray=oArray.copy() #iki kaynak birbirini etkilemez
vArray=oArray.view() #oArray i refere ettigi için birinde yapılan degişiklik ikisini de etkiler

print(cArray)
print(vArray)
print("-------------")
cArray[7]=999
print(oArray)
print(cArray)
print(vArray)
print("-----------------")
vArray[7]=111
print(oArray)
print(cArray)
print(vArray)

#2BOYUTLUYU 3BOYUTLUYA CEVİRME

array5=np.array([[1,2,3],
                 [4,5,6]])

print(array5.ndim, array5.shape)

nArray5=np.array([[1,2,3], [4,5,6]], ndmin=3)
print(nArray5.ndim, nArray5.shape)
print(nArray5)