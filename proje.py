#PROJE KİSMİ HATALİ YAY GEÇİDİ KULLANİMİ VE DURMAYAN TAŞİT
import cv2
import numpy as np
import time

# Fare ile alan seçimi için kullanılan değişkenler
drawing = False
points = []
areas = [] 

# Fare geri çağırma fonksiyonu
def draw_polygon(event, x, y, flags, param):
    global drawing, points

    if event == cv2.EVENT_LBUTTONDOWN:  # Sol tık ile nokta ekle
        drawing = True
        points.append((x, y))

    elif event == cv2.EVENT_LBUTTONUP:  # Sol tık bırakıldığında çizim bitir
        drawing = False

    elif event == cv2.EVENT_RBUTTONDOWN:  # Sağ tık ile işlemi sıfırla
        points = []

# Video açma
video = cv2.VideoCapture('Test.mp4')

alpha = 0.01  # Arka plan güncelleme hızı
field_threshold_yaya = 200
max_area_yaya = 250
area_threshold_tasit = 2000
max_distance_yaya = 94
max_distance_tasit = 120
yaya_sayisi = 0
tasit_sayisi = 0
speed_calculation_interval = 5  # Hizi 5 sn periyotlarla hesaplama

# İlk kareyi al ve kullanıcıdan maskeler oluşturmasını iste
ret, first_frame = video.read()
if not ret:
    print("Video okunamadi.")
    exit()

background = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY).astype("float")

cv2.namedWindow("Alan Maskesi Olustur")
cv2.setMouseCallback("Alan Maskesi Olustur", draw_polygon)

print("İstediginiz kadar alan maskesi olusturmak için sol tiklayin ve alanlari secin.")
print("Her bir alani tamamladiktan sonra 'r' tuşuna basarak alani kaydedin.")
print("İslem tamamlandiginda 'q' tusuna basarak devam edin.")

while True:
    temp_frame = first_frame.copy()

    # Kullanıcının seçtiği noktaları çizin
    if len(points) > 1:
        for i in range(len(points) - 1):
            cv2.line(temp_frame, points[i], points[i + 1], (0, 255, 0), 2)
        if len(points) > 2:
            cv2.line(temp_frame, points[-1], points[0], (0, 255, 0), 2)

    cv2.imshow("Alan Maskesi Olustur", temp_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):  # 'r' tuşu ile kaydet
        if len(points) > 2: 
            mask = np.zeros((first_frame.shape[0], first_frame.shape[1]), dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(points)], 255)
            areas.append(mask)
            points = []  # Yeni alan için noktaları sıfırla
            print(f"{len(areas)}. alan kaydedildi. Yeni bir alan cizmeye baslayabilirsiniz.")
        else:
            print("Gecerli bir alan tanimlanmadi. Alan için en az 3 nokta gerekiyor.")

    elif key == ord("q"):  # 'q' tuşu ile  bitir
        if len(points) > 0:
            print("Lütfen önce mevcut alani kaydetmek için 'r' tusuna basin.")
        else:
            print("Alan secim islemi tamamlandi.")
            break

    elif key == ord("c"):  
        points = []
        print("Mevcut alan secim sifirlandi. Yeni alan cizmeye baslayabilirsiniz.")

print(f"Toplam {len(areas)} alan maskesi olusturuldu.")
cv2.destroyWindow("Alan Maskesi Olustur")

yaya_gecidi_maskeleri = areas[:10] 

detected_centers_yaya = []
detected_centers_tasit = []
last_detection_time_yaya = []
last_detection_time_tasit = []
already_counted_yaya = []
already_counted_tasit = []
last_speed_calculation_time = time.time()  

kernel = np.ones((5, 5), np.uint8)

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Arka plan güncelleme ve maske oluşturma
    cv2.accumulateWeighted(gray_frame, background, alpha)
    background_frame = cv2.convertScaleAbs(background)
    frame_diff = cv2.absdiff(background_frame, gray_frame)
    _, fgmask = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)

    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    current_time = time.time()

    yaya_gecidi_dolu = False  # Yaya geçidinde yaya var mı 

    for contour in contours:
        area = cv2.contourArea(contour)
        (x, y, w, h) = cv2.boundingRect(contour)
        center = (x + w // 2, y + h // 2)

        if field_threshold_yaya < area < max_area_yaya:  # Yaya tespiti
            for yaya_gecidi_mask in yaya_gecidi_maskeleri:
                if yaya_gecidi_mask[y, x] == 255:
                    yaya_gecidi_dolu = True  # Yaya geçidinde yaya var
                    break

            color = (0, 255, 0) if yaya_gecidi_dolu else (0, 0, 255)
            label = "Yaya" if yaya_gecidi_dolu else "Yaya Gecit Disinda"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        elif area > area_threshold_tasit:  # Taşıt tespiti
            durmayan_tasit = False
            for yaya_gecidi_mask in yaya_gecidi_maskeleri:
                if yaya_gecidi_mask[y, x] == 255 and yaya_gecidi_dolu:
                    durmayan_tasit = True  # Yaya geçidinde durmayan taşıt

            color = (0, 0, 255) if durmayan_tasit else (255, 0, 0)
            label = "Durmayan Tasit" if durmayan_tasit else "Tasit"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imshow("Dört Yol Yaya ve Taşit İzleme", frame)
    slow_factor = 2  #  Video Yavaşlatma 
    delay = int(slow_factor * (1000 / video.get(cv2.CAP_PROP_FPS)))

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
