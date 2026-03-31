#VİZE KİSMİ ALGİLAMA+SAYMA+HİZ HESAPLAMA
import cv2
import numpy as np
import time

# Video kaynağını aç
video = cv2.VideoCapture('Train.mp4')

# Parametreler
alpha = 0.01  # Arka plan güncelleme hızı
field_threshold_yaya = 200
max_area_yaya = 250
area_threshold_tasit = 2000
max_distance_yaya = 94
max_distance_tasit = 120
reset_time = 3
min_visible_frames = 1
yaya_sayisi = 0
tasit_sayisi = 0
speed_calculation_interval = 5  # Hızın her 5 saniyede bir hesaplanması için süre

# İlk kareyi arka plan olarak al
ret, first_frame = video.read()
background = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY).astype("float")

# Yaya geçidi gibi statik alanların maskesini oluştur
yaya_gecidi_mask = np.zeros((first_frame.shape[0], first_frame.shape[1]), dtype=np.uint8)
cv2.rectangle(yaya_gecidi_mask, (100, 300), (540, 340), 255, -1)

detected_centers_yaya = []
detected_centers_tasit = []
last_detection_time_yaya = []
last_detection_time_tasit = []
visible_frames_yaya = []
visible_frames_tasit = []
already_counted_yaya = []
already_counted_tasit = []
last_speed_calculation_time = time.time()  # İlk hız hesaplama zamanını ayarla

# Morfolojik işlemler için kernel
kernel = np.ones((5, 5), np.uint8)

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    yaya_gecidi_mask_resized = cv2.resize(yaya_gecidi_mask, (frame.shape[1], frame.shape[0]))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Arka planı güncelle
    cv2.accumulateWeighted(gray_frame, background, alpha)
    background_frame = cv2.convertScaleAbs(background)

    # Arka plan çıkarma
    frame_diff = cv2.absdiff(background_frame, gray_frame)
    _, fgmask = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)

    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.bitwise_and(fgmask, fgmask, mask=cv2.bitwise_not(yaya_gecidi_mask_resized))

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    current_time = time.time()

    # Yayaları takip etmek için her yeni konturu kontrol et


    for contour in contours:
        area = cv2.contourArea(contour)
        if area < field_threshold_yaya or area > max_area_yaya:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        center = (x + w // 2, y + h // 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Yaya", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        is_new_yaya = True
        for i, detected_center in enumerate(detected_centers_yaya):
            distance = np.linalg.norm(np.array(center) - np.array(detected_center))
            if distance < max_distance_yaya:
                time_diff = current_time - last_detection_time_yaya[i]
                detected_centers_yaya[i] = center
                last_detection_time_yaya[i] = current_time
                visible_frames_yaya[i] += 1
                is_new_yaya = False

                # Eğer bu nesne daha önce sayılmadıysa say
                if not already_counted_yaya[i] and visible_frames_yaya[i] >= min_visible_frames:
                    yaya_sayisi += 1
                    already_counted_yaya[i] = True
                break

        if is_new_yaya:
            detected_centers_yaya.append(center)
            last_detection_time_yaya.append(current_time)
            visible_frames_yaya.append(1)
            already_counted_yaya.append(False)

    # Taşıtları takip etmek için her yeni konturu kontrol et
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < area_threshold_tasit:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        center = (x + w // 2, y + h // 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Tasit", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        is_new_tasit = True
        for i, detected_center in enumerate(detected_centers_tasit):
            distance = np.linalg.norm(np.array(center) - np.array(detected_center))
            if distance < max_distance_tasit:
                time_diff = current_time - last_detection_time_tasit[i]
                detected_centers_tasit[i] = center
                last_detection_time_tasit[i] = current_time
                visible_frames_tasit[i] += 1
                is_new_tasit = False

                if not already_counted_tasit[i] and visible_frames_tasit[i] >= min_visible_frames:
                    tasit_sayisi += 1
                    already_counted_tasit[i] = True
                break

        if is_new_tasit:
            detected_centers_tasit.append(center)
            last_detection_time_tasit.append(current_time)
            visible_frames_tasit.append(1)
            already_counted_tasit.append(False)

    # Her 5 saniyede bir hız hesaplama ve terminale yazdırma
    if current_time - last_speed_calculation_time >= speed_calculation_interval:
        print(f"[Zaman: {time.strftime('%H:%M:%S')}] Hız Hesaplama:")
        for i, center in enumerate(detected_centers_yaya):
            if visible_frames_yaya[i] >= min_visible_frames:
                time_diff = current_time - last_detection_time_yaya[i]
                speed = max_distance_yaya / time_diff if time_diff > 0 else 0
                print(f"ID: {i} | Tür: Yaya  | Hız: {speed:.2f} px/sn")
        
        for i, center in enumerate(detected_centers_tasit):
            if visible_frames_tasit[i] >= min_visible_frames:
                time_diff = current_time - last_detection_time_tasit[i]
                speed = max_distance_tasit / time_diff if time_diff > 0 else 0
                print(f"ID: {i} | Tür: Taşıt | Hız: {speed:.2f} px/sn")
        
        last_speed_calculation_time = current_time  # Zamanlayıcıyı sıfırla

    cv2.imshow("Dört Yol Yaya ve Taşit İzleme", frame)
    delay = int(1000 / video.get(cv2.CAP_PROP_FPS))
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

print(f"Toplam Yaya Sayisi: {yaya_sayisi}")
print(f"Toplam Taşit Sayisi: {tasit_sayisi}")
video.release()
cv2.destroyAllWindows()