# 04 – Deney ve Değerlendirme Planı

## 1. Metrikler

Hepsi test senaryoları üzerinden hesaplanır, senaryolar arası ortalama ± std verilir.

| Metrik | Tanım | Yön |
|---|---|---|
| RMSE | `sqrt(mean((T_true − T_ref)²))`, tüm bölgeler ve adımlar | ↓ |
| IAE | `Σ |T_true − T_ref| · Δt` | ↓ |
| Maks. düşüş | Parti girişinden sonraki en büyük `T_ref − T_true` (°C) | ↓ |
| Toparlanma süresi | Düşüşten sonra ±5 °C bandına dönüp orada kalana kadar geçen süre (dk) | ↓ |
| Aşma | En büyük `T_true − T_ref` (°C) | ↓ |
| Toplam enerji | Bölüm başına kWh; PID'e göre % fark | ↓ |
| Eylem titreşimi | `mean |u_t − u_{t−1}|` | ↓ |
| Güvenlik ihlali | `T > T_max` olan bölüm sayısı | 0 olmalı |
| Toplam ödül | Bölüm başına | ↑ |
| (Ek) Parti çıkış sıcaklığı | Fırından çıkan partinin hedefe yakınlığı (kalite göstergesi) | — |

## 2. Baseline: bölge başına PID

- Ayrık PID: `u = Kp·e + Ki·Σe·Δt + Kd·Δe/Δt`, `e = T_ref − T_meas`
- Çıkış `[0, 1]`'e kırpılır; **anti-windup** (clamping ya da back-calculation)
- Türev terimi ölçüm üzerinden ve alçak geçiren filtreyle (gürültüye karşı)
- **Ayarlama (E1, `scripts/tune_pid.py`):** 10 ayar senaryosunda (test setinden farklı seed'ler)
  `Kp × Ti × Td = 6 × 5 × 3 = 90` kombinasyonluk ızgara araması. Amaç RL ile aynı ödül fonksiyonu.
  Böylece kıyas adil olur ve "PID'i kötü ayarlayıp RL'i iyi gösterme" eleştirisi önlenir.
- **Kararlılık şartı:** ödül yüzeyi kazançlara karşı düz olduğu için yalnızca ödüle bakan arama,
  yük yokken bile ±7–14 °C salınan ayarları seçiyordu (sahada kabul edilmez). Bu yüzden önce şart
  uygulanır: yüksüz, hedefin 30 °C altından başlayan adım yanıtında son 30 dk tepe-tepe salınım
  < 2 °C. 90 ayardan 21'i şartı sağlıyor. Seçilen: **Kp = 0.04 /°C, Ti = 300 s, Td = 60 s**
  (`configs/pid.yaml`).
- (Yapılmadı, gelecek çalışma) PID + kütle tabanlı ileri besleme (feedforward). E8'e göre RL'in kazancı önizleme
  bilgisinden geldiği için bu baseline farkı önemli ölçüde küçültebilir.

## 3. Q-tablosu ayarları (referans: `configs/q_table.yaml`)

Derin RL kullanılmaz. Derste işlenen tablolu Q-öğrenme kullanılır (`src/furnace_rl/agents/q_table.py`).
Her bölgenin kendi Q-tablosu vardır; güncelleme `Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]`.

| Hiperparametre | Değer |
|---|---|
| Durum | 10×10 ızgara: sıcaklık hatası (10 kutu) × yük durumu (10 sınıf) |
| Hata kutu sınırları | −100, −60, −30, −15, −8, −3, 3, 8, 20 °C |
| Yük sınıfları | yük yok (sıcaklık düşüyor / sabit / yükseliyor), küçük / büyük parti yaklaşıyor, hafif yük (2), ağır yük (2), yük var ve büyük parti yaklaşıyor |
| Eylem | güç seviyesi ∈ {0, 0.15, 0.25, 0.35, 0.5, 1.0}, 60 s'de bir karar |
| Keşif | ε-greedy, ε 0.3 → 0.02 (4000 bölümde doğrusal) |
| Öğrenme oranı | α(s,a) = max(0.02, 0.5 / (1 + ziyaret(s,a) / 20)) |
| γ | 0.99 (kontrol adımı başına) |
| Başlangıç Q | −5 |
| Toplam | 5000 bölüm, 3 600 000 adım |
| Değerlendirme | her 50 bölümde 5 doğrulama senaryosu, en yüksek ödüllü tablo kaydedilir |
| Seed | 0, 1, 2 |

**Izgara seçimi (doğrulama senaryoları, seed 0).** Test senaryoları kullanılmadan, aşağıdaki sırayla seçildi:

| Durum temsili | Hücre (bölge başına) | En yüksek doğrulama ödülündeki RMSE [°C] |
|---|---|---|
| 11 hata × 5 hız × 4 yük × 4 yaklaşan yük, 11 güç seviyesi | 9 680 | 73.5 |
| 14 hata × 5 × 4 × 4, ziyarete göre azalan α | 12 320 | 34.7 |
| 14 hata × 5 × 3 × 3, 10 000 bölüm | 6 930 | 24.8 |
| **10 hata × 10 yük durumu, 6 güç seviyesi** | **600** | **13.3** |
| 10 hata × 10 değişim hızı (yük ve önizleme yok) | 600 | 22.3 |
| PID (karşılaştırma) | – | 11.9 |

Tablo küçüldükçe her hücre daha sık güncellendiği için öğrenme hızlandı. Eğitimde izleme cezasını sınırlamak
(err_clip) denendi ve sonucu kötüleştirdi, kullanılmadı.

**Son ayar (`configs/q_table_son.yaml`, ortam `configs/env_w_energy_1.0.yaml`).** E4 ve E5'in en iyi iki bulgusu
(güç seviyeleri 0/0.2/0.3/0.4/0.6/1 ve w₂ = 1) birleştirildi, 30 s karar periyodu eklendi. Üç seed'in doğrulama
RMSE ortalaması: referans 14.31, seviyeler + w₂ = 1 14.15, son ayar 13.75 °C. En düşük olan son ayar seçildi,
test senaryoları bu seçimden sonra çalıştırıldı. E3, E6, E7 son ayarla; E4, E5, E8 referans ayarla yapıldı.
Ödül, test sırasında tüm politikalar için varsayılan ağırlıklarla (w₂ = 0.2) hesaplanır.

## 4. Deney listesi

| ID | Deney | Amaç | Sprint |
|---|---|---|---|
| E0 | Simülatör kalibrasyonu: sabit güç, rastgele politika, PID | Ortamın gerçekçi ve öğrenilebilir olduğunu göstermek | 1 |
| E1 | PID ızgara araması | Adil baseline | 1 |
| E2 | Q-tablosu – 3 seed eğitim | Öğrenme eğrisi (ortalama ± std), kararlılık | 2 |
| E3 | PID ve Q-tablosu – nominal test seti (20 senaryo) | Ana kıyas tablosu ve grafikleri | 2 |
| E4 | Ödül ağırlığı taraması `w₂ ∈ {0, 0.2, 1}`, `w₃ = 0` | Reward shaping yan etkileri, RMSE–kWh | 3 |
| E5 | Hiperparametre analizi (güç seviyeleri, α_min, ε, ızgaranın ikinci ekseni, bulguların birleşimi), 3 seed | Duyarlılık, son ayarın seçimi | 2–3 |
| E6 | Sensör gürültüsü σ ∈ {0, 1, 3, 5} °C | Gürbüzlük (rubrikte açıkça isteniyor) | 3 |
| E7 | Dağılım dışı senaryolar: 2500 kg parti, 0.05 / 0.4 m/dk bant, 8 dk arayla iki parti | Genelleme | 3 |
| E8 | Önizleme ablasyonu: gözlemde `m_next, v_next, t_arrival` yok | Önizleme bilgisinin katkısı | 3 |

## 5. Grafikler (rapor için)

1. Öğrenme eğrisi: adım – bölüm ödülü, 3 seed ortalaması ve gölgeli std bandı
2. Örnek senaryo zaman serisi: 3 bölge sıcaklığı (PID ve Q-tablosu üst üste, hedef kesikli çizgi),
   parti girişleri dikey çizgiyle işaretli
3. Aynı senaryoda güç komutları `u_i(t)`, PID ve Q-tablosu
4. Kıyas tablosu ve çubuk grafik: RMSE, maks. düşüş, kWh
5. E4 Pareto: x = kWh, y = RMSE, her nokta bir `w₂`
6. E6 gürültü: x = σ_noise, y = RMSE (PID ve Q-tablosu)
7. E8 önizleme ablasyonu: çubuk grafik

## 6. Çıktı düzeni

```
results/
├── models/q_son_seed{0,1,2}/best_model.npz   # son ayar (depoda)
├── logs/          # eğitim ve değerlendirme çıktıları
├── eval/          # her deney için metrics.csv
└── figures/       # rapora girecek .png dosyaları
```
