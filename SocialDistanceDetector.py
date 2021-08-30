import numpy as np
import imutils
import time
import cv2
import os
import math
import Person

from itertools import chain #itt bilg daha verimli kullanmak için,döngü için,
from constants import *
#YOLOv3 farklı düzenlerde eğitilmiş modellere sahiptir. Veriseti olarak COCO kullanılmıştır. Eğitilmiş modeller ile 80 tane nesne tespit edilebilir.

LABELS = open(YOLOV3_LABELS_PATH).read().strip().split('\n')#Daha onceden egitilen yolo modeli yuklenir

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype='uint8')

print('Loading YOLO from disk...')

neural_net = cv2.dnn.readNetFromDarknet(YOLOV3_CFG_PATH, YOLOV3_WEIGHTS_PATH)  #yolo yuklemesi
layer_names = neural_net.getLayerNames() # layer_names , YOLO'dan ihtiyacımız olan tüm çıktı katmanı adlarını içerir
layer_names = [layer_names[i[0] - 1] for i in neural_net.getUnconnectedOutLayers()]

vs = cv2.VideoCapture(VIDEO_PATH) # videonun yolu constants.py de belirtilmiştir.
writer = None
(W, H) = (None, None)#video boyutunu yok sayar

cnt_up = 0 #giriş sayacı
cnt_down = 0 # çıkış sayacı
#arkaplanın temizlenmesi işlemi
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

#kernellar olusturuldu.
kernelOp = np.ones((3, 3), np.uint8)
kernelOp2 = np.ones((5, 5), np.uint8)
kernelCl = np.ones((11, 11), np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX
persons = []# boş persons değişkeni
max_p_age = 5
pid = 1
sayac=1
sayac2=1
sonuc=0
down=0
sosyal=0
deg=0
def convertBack(x, y, w, h):

    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax
try:
    if(imutils.is_cv2()):
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT #video akışındaki kare sayısını saymak için
    else:
        prop = cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print('Tespit edilen toplam frame sayısı ', total)

except Exception as e:
    print(e)
    total = -1

while True:
    (grabbed, frame) = vs.read() #videonun okunması

    if not grabbed: #videodan görüntü alınamadıysa break edilir.
        break
    #H ve W değişkenlerimizi (None, None) 'dan (height_of_frame, width_of_frame) olarak güncelliyoruz
    if W is None or H is None:
        H, W = (frame.shape[0], frame.shape[1])#framein  w ve h değeri atanır
    #R&B renk dizilerini değiştirmek için swapRB = True argümanını geçiyoruz
    #dizi öğelerini 255'e bölerek görüntüyü yeniden ölçeklendiriyoruz, böylece her öğe 0 ile 1 arasında yer alır
    #daha iyi bir performans sergilemek için.

    frameArea = H * W  # ekran boyutu
    areaTH = 500
    line_up = int(2 * (H / 5))  # mavi cizgi.
    line_down = int(3 * (H / 5))  # kırmızı çizgi

    up_limit = int(1 * (H/ 5))  # alt beyaz çizgi
    down_limit = int(4 * (H / 5))  # üst beyaz cizgi

    '''print("Red line y:", str(line_down))
    print("Blue line y:", str(line_up))
    print("up limit y:", str(up_limit))
    print("down limit y:", str(down_limit))'''

    line_down_color = (0, 0, 255)  # kırmızı renk
    line_up_color = (255, 0, 0)  # mavi renk

    pt1 = [0, line_down];
    pt2 = [W, line_down];
    pts_L1 = np.array([pt1, pt2], np.int32)
    pts_L1 = pts_L1.reshape((-1, 1, 2))  # kırmızı yani çıkanlar. down artar

    pt3 = [0, line_up];
    pt4 = [W, line_up];
    pts_L2 = np.array([pt3, pt4], np.int32)
    pts_L2 = pts_L2.reshape((-1, 1, 2))  # mavi çizgi yani girenler. up artar.

    pt5 = [0, up_limit];
    pt6 = [W, up_limit];
    pts_L3 = np.array([pt5, pt6], np.int32)
    pts_L3 = pts_L3.reshape((-1, 1, 2))  # üst beyaz çizgi

    pt7 = [0, down_limit];
    pt8 = [W, down_limit];
    pts_L4 = np.array([pt7, pt8], np.int32)
    pts_L4 = pts_L4.reshape((-1, 1, 2))  # alt beyaz çizgi

    fgmask = fgbg.apply(frame)  # maskeleme işlemin yapılması
    fgmask2 = fgbg.apply(frame)

    try:
        ret, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)  # 200 255 arasında değerleri döndürür.

        # Opening
        mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)

        # Closing
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelCl)

    except:
        print('EOF')
        print('UP:', cnt_up)
        print('DOWN:', cnt_down)
        break

    contours0, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours0:
        area = cv2.contourArea(cnt)
        centroid_dict = dict()
        objectId = 0

        if area > areaTH:

            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])  # ağırlıklı merkezin x bileşeni
            cy = int(M['m01'] / M['m00'])  # ağırlıklı merkezin Y bileşeni

            x, y, w, h = cv2.boundingRect(cnt)  # x,y başlangıc noktaları koordinatlar w genişlik h yükseklik

            new = True
            # frame e girdiği andan itibaren.
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 255,0,0 mavi 0 0 0 siyah 0, 255, 0 BGR*
            # frame girildiği anda direk kırmızı rectangledır.

            if cy in range(up_limit,down_limit):  # alt beyaz çizgi ile üst beyaz cizgi arasında ise benm cy ağırlık merkezim ?

                for i in persons:

                    xmin = int(round(x - (w / 2)))
                    xmax = int(round(x + (w / 2)))
                    ymin = int(round(y - (h / 2)))
                    ymax = int(round(y + (h / 2)))
                    xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
                    centroid_dict[objectId] = (int(x), int(y), x, y, x + w, y + h)  # çözüldü.
                    objectId += 1
                    # abs mutlak değer alır.

                    if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h:
                        # print("Red line x:", str(x))
                        # print("Red line i.getX():", str(i.getX()))
                        # print("Red line w:", str(w))
                        # print("Red line y:", str(y))
                        # print("Red line i.gety():", str(i.getY()))
                        # print("Red line h:", str(h))
                        # abs mutlak değer alır.
                        new = False
                        i.updateCoords(cx, cy)
                        if i.going_UP(line_down, line_up) == True:  # state= 1 & dir= up
                            print("sayacccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc arttı. UP")
                            cv2.line(frame, (x, y), (120, 40), (255,0,0), 5)  # 50      3
                            cnt_up += 1;

                        elif i.going_DOWN(line_down, line_up) == True:  # state= 1 & dir= down
                            print("sayaccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc arttı. DOWN")
                            cv2.line(frame, (x, y), (120, 70), (0, 0, 255), 5)  # 50      3
                            cnt_down += 1;

                        break
                    if i.getState() == '1':
                        if i.getDir() == 'down' and i.getY() > down_limit:
                            i.setDone()  # self.done = true  olmus olucak
                        elif i.getDir() == 'up' and i.getY() < up_limit:
                            i.setDone()  # self.done = true
                if new == True:
                    p = Person.MyPerson(pid, cx, cy, max_p_age)
                    persons.append(p)
                    pid += 1
                # print("xmin ", xmin)

            #cv2.circle(frame, (cx, cy), 5, (255, 255, 255),-1)  # ağırlık merkezini yuvarlark olarak frame üzerinde gösteririr.
        # img = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) #insanın üzerinde rectangle cizer.
        # cv.drawContours(frame, cnt, -1, (0,255,0), 3)
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    neural_net.setInput(blob)

    start_time = time.time()
    layer_outputs = neural_net.forward(layer_names)
    end_time = time.time()
    
    boxes = []
    confidences = []
    classIDs = []
    lines = []
    box_centers = []

    for output in layer_outputs:#her çıktının
        for detection in output:#her çıktıdaki her algılamanın
            
            scores = detection[5:] #6değer vardır.
            #print("detection1", detection[0:4])
            classID = np.argmax(scores)
            confidence = scores[classID]
            #print("confi",confidences)#1 nesne ne kadar insan olma ihtimali?

            if confidence > 0.3 and classID == 0:#eşik güvenirliğini(güven aralığı) 0,5 olarak ayarladık.
                #print("detection1", detection[0:4])
                #w = 640 h = 480
                #detection [0.09132846 0.43206498 0.099273   0.26022345]
                # box [ 58.45021248 207.39119053  63.53471756 124.90725517]
                box = detection[0:4] * np.array([W, H, W, H]) #w genişlik h yükseklik
                #print("box",box)
                #print("w, h",W,H) #640 480 dir.
                #print("detection",detection[0:4])#ilk 4 değer elde edilir.

                (centerX, centerY, width, height) = box.astype('int')#58.45021248 olan değer 58 olarak elde edilir.
                print("centerX", centerX)
                x = int(centerX - (width / 2)) # rectanglenın sol üst kösesinin x koordinatı bulunur.
                y = int(centerY - (height / 2)) # rectanglenın sol üst kösesinin y koordinatı bulunur.

                #cv2.circle(frame, (centerX, centerY), 5, (250, 0, 0), -1)
                #cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

                box_centers = [centerX, centerY]#orta noktalar atılıyor
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                #print("confidences",confidence)
    #cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
    #cv2.putText(frame, '.', (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 255, 255), 4, cv2.LINE_4)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    print("idxs",idxs)
    # boxes, confidences değerleri varsa; demekki nesneyi insan olarak algılamıstır. ve algılama sonucunda boxes a ve confidences a atama yapılmıştır.
    # bu atama yapıldıysa idxs [[0]] olur ve len(idxs) değeri 1 olmaktadır. YANİ SEN İNSANSIN.
    if len(idxs) > 0:
        print("len(idxs)",len(idxs))
        unsafe = []
        count = 0
        count2 = 0
        for i in idxs.flatten():
            
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            centeriX = boxes[i][0] + (boxes[i][2] // 2)
            centeriY = boxes[i][1] + (boxes[i][3] // 2)
            #cv2.circle(frame, (150, centeriY), 5, (0, 0, 0), -1) #siyah.
            color = [int(c) for c in COLORS[classIDs[i]]]
            text = '{}: {:.3f}'.format(LABELS[classIDs[i]], confidences[i]) #nesnenin insan olma ihtimalinin frame üzerinde göstermek için kulllandık.
            if centeriY< up_limit:
                katsayi1=2.4
            if centeriY>up_limit and centeriY<line_up:
                katsayi1=2.03
            if centeriY > line_up and centeriY < line_down:
                katsayi1=1.76
            if centeriY > line_down and centeriY < W:
                katsayi1=1.36
            idxs_copy = list(idxs.flatten())
            idxs_copy.remove(i)
            sel=line_down-line_up
            print("line_down line_up sel",line_down,line_up,sel)
            distance2 = math.sqrt(line_up**2 + line_down**2)
            distance3 = math.sqrt(math.pow(line_down,2) + math.pow(line_up,2))
            distance4 = math.sqrt(math.pow(0 - 0, 2) + math.pow(432 - 288, 2))
            katsayi=distance3/sel
            print("distance2", distance2,distance4)


            for j in np.array(idxs_copy):
                centerjX = boxes[j][0] + (boxes[j][2] // 2)
                centerjY = boxes[j][1] + (boxes[j][3] // 2)

                distance = math.sqrt(math.pow(centerjX - centeriX, 2) + math.pow(centerjY - centeriY, 2))
                Distance=distance/katsayi
                distance=distance*katsayi1 #cm cinsine cevrilmiştir.
                #sosyal mesafe 86 pixel ve 150 cmdir.
                if distance <= SAFE_DISTANCE:
                    cv2.line(frame, (boxes[i][0] + (boxes[i][2] // 2), boxes[i][1]  + (boxes[i][3] // 2)), (boxes[j][0] + (boxes[j][2] // 2), boxes[j][1] + (boxes[j][3] // 2)), (0, 0, 255), 2)
                    cv2.putText(frame, 'Mesafe : {:.1f} cm'.format(distance), (x,y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,160), 2)
                    sayac2 = sayac2 + 1
                    print("Aralarindaki  Uzaklik",distance)
                    unsafe.append([centerjX, centerjY])
                    unsafe.append([centeriX, centeriY])
                    #Sosyal mesafeye UYANLAR da arasındaki mesafeyi göster.
                if distance > SAFE_DISTANCE and distance<300:
                    sayac = sayac + 1
                #print("sayac2", sayac2,sayac)
                sonuc = sayac2 / sayac
                str_sonuc = 'sonuc:' + str(sonuc)
                #print("sayac2", sayac2)a
                #cv2.putText(frame, str_sonuc, (10, 120), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)



                '''if distance > SAFE_DISTANCE:
                    cv2.line(frame, (boxes[i][0] + (boxes[i][2] // 2), boxes[i][1] + (boxes[i][3] // 2)),(boxes[j][0] + (boxes[j][2] // 2), boxes[j][1] + (boxes[j][3] // 2)), (0, 255, 0), 2)
                    cv2.putText(frame, 'Aralarindaki Mesafe {:.3f}'.format(distance), (x, y - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
'''


            if centeriX in chain(*unsafe) and centeriY in chain(*unsafe):
                count += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            else:
                count2 += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (79, 79, 47), 1) #insan mı değil mi? güven aralıgı. 0,95448  & 0,85465 gibi yazan ifade.
            cv2.rectangle(frame, (0, down_limit+30), (int(W/3), down_limit+75),(240, 250, 120), -1) #üst üste yazıdıgı için görünülürlüğünü arttırmak için.*************************
            cv2.putText(frame, 'Kisi Sayisi = {}'.format(count2 + count), (0, down_limit+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 250), 2)#
            print("dow00",down_limit)

           # cv2.rectangle(frame, (442, 35), (575, 17), (0, 0, 0), -1)#femx yazı kapanmasıdır.
            cv2.rectangle(frame, (0, down_limit + 70), (int(W/3), down_limit + 110), (240, 250, 120), -1)
            cv2.putText(frame, 'Sosyal Mesafeye Uymayan Insan sayisi = {}'.format(count), (0, down_limit+75),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 250), 2)  #

            cv2.putText(frame, 'Insanlarin % {:.2f} Sosyal Mesafeye Uymamaktadir.'.format(float(count/(count2+count))*100), (0, down_limit + 100),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 250), 2)
            print('Insanlarin % {:.2f} Sosyal Mesafeye Uymamaktadir.'.format(float(count/(count2+count))*100))
            deg+=float(count/(count2+count))*100
            sosyal+=1
            deg2=int(deg)
            print("deg", deg2, sosyal)
            total-=1
            print("total",total)
            if sosyal <2000:
                ortanokta = (line_down + down_limit) // 2

                cv2.line(frame, (200, line_down), (200, down_limit + 20), (240, 250, 120), 5)  # 50      3
                cv2.line(frame, (150, ortanokta), (200, down_limit + 20), (240, 250, 120), 5)  # 50      3
                cv2.line(frame, (250, ortanokta), (200, down_limit + 20), (240, 250, 120), 5)  # 50      3
            if sosyal > 4000 and sosyal <6000:
                ortanokta = (line_down + down_limit) // 2

                cv2.line(frame, (200, line_down), (200, down_limit + 20), (240, 250, 120), 5)  # 50      3
                cv2.line(frame, (150, ortanokta), (200, down_limit + 20), (240, 250, 120), 5)  # 50      3
                cv2.line(frame, (250, ortanokta), (200, down_limit + 20), (240, 250, 120), 5)  # 50      3
            if sosyal > 8000 and sosyal <10000:
                ortanokta = (line_down + down_limit) // 2

                cv2.line(frame, (200, line_down), (200, down_limit + 20), (240, 250, 120), 5)  # 50      3
                cv2.line(frame, (150, ortanokta), (200, down_limit + 20), (240, 250, 120), 5)  # 50      3
                cv2.line(frame, (250, ortanokta), (200, down_limit + 20), (240, 250, 120), 5)  # 50      3#123, 45, 78 mor renktir.

           # if sosyal > 10000: #video5 için sağ alttaki kısımdır.
            #    ortanokta = (line_down + down_limit) // 2

             #   cv2.line(frame, (820, line_down), (820, down_limit + 20), (240, 250, 120), 5)  # 50      3
              #  cv2.line(frame, (770, ortanokta), (820, down_limit + 20), (240, 250, 120), 5)  # 50      3
               # cv2.line(frame, (870, ortanokta), (820, down_limit + 20), (240, 250, 120), 5)  # 50      3

               # cv2.rectangle(frame, (int(w/2)+625, 600), (int(w/2)+1000, down_limit+60), (240, 0, 120), -1)
               # cv2.putText(frame, 'Insanlarin Sosyal Mesafeye Uyma Orani % {:.2f} '.format(100-float(deg2/sosyal) ), (int(W/2), down_limit+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 250, 0), 2)
#kbb için sağ altta cıkacak ifadenin yazılımı.
            if sosyal > 11600:
                ortanokta = (line_down + down_limit) // 2

                cv2.line(frame, (870, line_down), (870, down_limit + 20), (240, 250, 120), 5)  # 50      3
                cv2.line(frame, (820, ortanokta), (870, down_limit + 20), (240, 250, 120), 5)  # 50      3
                cv2.line(frame, (920, ortanokta), (870, down_limit + 20), (240, 250, 120), 5)  # 50      3

                cv2.rectangle(frame, (int(w/2)+655, 640), (int(w/2)+1060, down_limit+60), (240, 0, 120), -1)
                cv2.putText(frame, 'Insanlarin Sosyal Mesafeye Uyma Orani % {:.2f} '.format(100-float(deg2/sosyal) ), (int(W/2), down_limit+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 250, 0), 2)

#2 beyaz çizgi arasındaki alana girdikleri anda id numarası alınır. ve bu ıd numarası frame üzerinde gösterilmesidir.
   # for i in persons: # frame üzerinde insanların sayısını yazar. üzerlerinde belirtir yani

      #  cv2.putText(frame, str(i.getId()), (i.getX(), i.getY()), font, 0.4, i.getRGB(), 1, cv2.LINE_AA) #


    if writer is None:

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30,(frame.shape[1], frame.shape[0]), True)

        if total > 0:
            elap = (end_time - start_time)
            print('1 frame işlenme süresi {:.4f} '.format(elap))
            print('Tahmini total süre: {:.4f} seconds'.format(elap * total))

    str_up = '    UP = ' + str(cnt_up)
    str_down = ' DOWN = ' + str(cnt_down)

    frame = cv2.polylines(frame, [pts_L1], False, line_down_color, thickness=2)  # kırmızı yer yani çıkanlar.
    frame = cv2.polylines(frame, [pts_L2], False, line_up_color, thickness=2)
    frame = cv2.polylines(frame, [pts_L3], False, (255, 255, 255), thickness=1)  # üst beyaz çizgi
    frame = cv2.polylines(frame, [pts_L4], False, (255, 255, 255), thickness=1)  # alt beyaz çizgi


    #cv2.line(frame, (int(W/2), line_up), (int(W/2), line_down), (123, 45, 78), 3)
    print("lineup, down",line_up,line_down)
    #lineup, down 432 648 değerleridir.
    #frame = cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # alt beyaz çizgi

    cv2.putText(frame, str_up, (10, 40), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, str_down, (10, 70), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)



    # cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
   # cv2.putText(frame, '.asdasdas', (x, y), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    #cv2.line(frame, (285,60), (310,60), (123, 45, 78), 1)
    #referans DEĞERLERİ
    #150 santimetre 1 tanesi 75 santimetre olarak esas alınmıstır.

   # cv2.line(frame, (406, 60), (456, 60), (123, 45, 78), 2)#50      3
    #cv2.line(frame, (365, 180), (424, 180), (123, 45, 78), 2)#59    2,542
    #cv2.line(frame, (317, 310), (385, 310), (123, 45, 78), 2)#68    2,205
    #cv2.line(frame, (244, 504), (332, 504), (123, 45, 78), 2)#88 150/88=1,704

    writer.write(frame)
    cv2.imshow('Sosyal Mesafe', frame)
    k = cv2.waitKey(30) & 0xff#Esc çıkış
    if k == 27:
        break

print('Cleaning up...')
writer.release()
vs.release()

cv2.destroyAllWindows()

