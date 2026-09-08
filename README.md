# Gerçek Zamanlı Kavşak Tespit Algoritması

Video görüntüsü üzerinde yol kavşaklarını gerçek zamanlı olarak tespit
eden bir algoritma. Makine öğrenmesi kullanılmıyor — tespit tamamen
klasik görüntü işleme adımlarıyla yapılıyor.

## Neden makine öğrenmesi kullanılmadı

Derin öğrenme tabanlı yaklaşımlar etiketlenmiş büyük veri setleri ve
eğitim için hesaplama gücü gerektiriyor. Kavşak yapısı belirli geometrik
özelliklere (kesişen çizgiler, yol kenarı süreksizlikleri) sahip olduğu
için bu problem klasik yöntemlerle de çözülebiliyor. Bu yaklaşım eğitim
verisi gerektirmiyor ve düşük donanımda gerçek zamanlı çalışabiliyor.

## Nasıl çalışıyor

1. **Kare okuma** — video akışından kareler sırayla alınır
2. **Ön işleme** — gri tonlama, bulanıklaştırma, ONISLEME_ADIMLARIN
3. **Kenar tespiti** — KENAR_ALGORITMASI (örn. Canny) ile kenarlar çıkarılır
4. **İlgi alanı seçimi** — görüntünün yalnızca yol bölgesi maskelenir
5. **Çizgi tespiti** — CIZGI_ALGORITMASI (örn. Hough dönüşümü) ile yol
   çizgileri bulunur
6. **Kavşak kararı** — KARAR_KRITERI (örn. çizgi açılarındaki değişim,
   yol kenarındaki süreksizlik) değerlendirilerek kavşak olup olmadığına
   karar verilir ve kare üzerinde işaretlenir

## Dosyalar

| Dosya | İçerik |
|---|---|
| `proje.py` | Algoritmanın son hali — çalıştırılacak dosya budur |
| `vize.py` | Projenin ara aşamadaki ilk sürümü, referans olarak bırakıldı |
| `Test.mp4` | Algoritmanın denendiği örnek video |
| `Train.mp4` | İkinci örnek video |

**Not:** Video isimleri "Train" ve "Test" olsa da bu proje makine
öğrenmesi kullanmıyor. İkisi de yalnızca algoritmanın denendiği örnek
kayıtlardır, eğitim verisi değildir.

## Kurulum ve çalıştırma

```bash
pip install opencv-python numpy
python proje.py
```

Program video dosyasını kare kare işleyip tespit sonucunu ekranda
gösterir.

## Kısıtlar

- Aydınlatma ve hava koşullarındaki değişimlere duyarlı
- Yol çizgilerinin belirgin olduğu görüntülerde çalışıyor
- DIGER_KISIT

## Kullanılan kütüphaneler

OpenCV · NumPy
