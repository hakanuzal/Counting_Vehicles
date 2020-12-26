import cv2
import numpy as np

Video_Okuyucu = cv2.VideoCapture("arabavideo.mp4")
fgbg = cv2.createBackgroundSubtractorMOG2()

kernel = np.ones((5,5),np.uint8)

class Koordinat:
    def __init__(self,x,y):
        self.x=x
        self.y=y

class Sensör:
    def __init__(self,Koordinat1,Koordinat2,Kare_Genişlik,Kare_Uzunluk):
        self.Koordinat1 = Koordinat1
        self.Koordinat2 = Koordinat2
        self.Kare_Genişlik = Kare_Genişlik
        self.Kare_Uzunluk = Kare_Uzunluk
        self.Maskenin_Alanı = abs(self.Koordinat2.x-Koordinat1.x)*abs(self.Koordinat2.y-self.Koordinat1.y)
        self.Maske = np.zeros((Kare_Uzunluk,Kare_Genişlik,1),np.uint8)
        cv2.rectangle(self.Maske, (self.Koordinat1.x,self.Koordinat1.y),(self.Koordinat2.x,self.Koordinat2.y),(255),thickness=cv2.FILLED)
        self.Durum = False
        self.Algılanan_Araç_Sayısı = 0



Sensör1 = Sensör(Koordinat(310,180),Koordinat(420,240),1080,250)

font = cv2.FONT_HERSHEY_SIMPLEX


while (1):
    ret, Kare = Video_Okuyucu.read()

    Kesilmiş_Kare = Kare[350:600,100:1180]

    Arka_Plan_Silinmiş_Kare = fgbg.apply(Kesilmiş_Kare)
    Arka_Plan_Silinmiş_Kare = cv2.morphologyEx(Arka_Plan_Silinmiş_Kare, cv2.MORPH_OPEN, kernel)
    ret, Arka_Plan_Silinmiş_Kare = cv2.threshold(Arka_Plan_Silinmiş_Kare, 80, 255, cv2.THRESH_BINARY)

    cnts, hierarchy = cv2.findContours(Arka_Plan_Silinmiş_Kare, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    Sonuç = Kesilmiş_Kare.copy()

    Doldurulmuş_Resim = np.zeros((Kesilmiş_Kare.shape[0],Kesilmiş_Kare.shape[1],1),np.uint8)


    for cnt in cnts :
        x, y, w, h = cv2.boundingRect(cnt)
        if(w>30 and h>30):
            cv2.rectangle(Sonuç,(x,y),(x+w,y+h),(0,255,0),thickness=4)
            cv2.rectangle(Doldurulmuş_Resim,(x,y),(x+w,y+h),(255),thickness=cv2.FILLED)
    Sensör1_Maske_Sonuç = cv2.bitwise_and(Doldurulmuş_Resim,Doldurulmuş_Resim,mask=Sensör1.Maske)
    Sensör1_Beyaz_Piksel_Sayısı = np.sum(Sensör1_Maske_Sonuç==255)
    Sensör1_Oran = Sensör1_Beyaz_Piksel_Sayısı/Sensör1.Maskenin_Alanı

    if(Sensör1_Oran>=0.75 and Sensör1.Durum==False):
        cv2.rectangle(Sonuç, (Sensör1.Koordinat1.x, Sensör1.Koordinat1.y), (Sensör1.Koordinat2.x, Sensör1.Koordinat2.y),
                      (0, 255, 0), thickness=cv2.FILLED)
        Sensör1.Durum=True
    elif(Sensör1_Oran<=0.75 and Sensör1.Durum==True):
        cv2.rectangle(Sonuç, (Sensör1.Koordinat1.x, Sensör1.Koordinat1.y), (Sensör1.Koordinat2.x, Sensör1.Koordinat2.y),
                      (0, 0, 255), thickness=cv2.FILLED)
        Sensör1.Durum=False
        Sensör1.Algılanan_Araç_Sayısı+=1
    else:
        cv2.rectangle(Sonuç, (Sensör1.Koordinat1.x, Sensör1.Koordinat1.y), (Sensör1.Koordinat2.x, Sensör1.Koordinat2.y),
                      (0, 0, 255), thickness=cv2.FILLED)
    cv2.putText(Sonuç,str(Sensör1.Algılanan_Araç_Sayısı),(Sensör1.Koordinat1.x,Sensör1.Koordinat1.y+60),font,3,(255,255,255))


    #cv2.imshow("Kare",Kare)
    #cv2.imshow("Kesilmiş Kare",Kesilmiş_Kare)
    cv2.imshow("Arka Plan Silinmiş Kare", Arka_Plan_Silinmiş_Kare)
    #cv2.imshow("Doldurulmuş Resim",Doldurulmuş_Resim)
    cv2.imshow("Sonuç",Sonuç)


    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

Video_Okuyucu.release()
cv2.destroyAllWindows()


