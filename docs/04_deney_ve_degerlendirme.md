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

## 3. MB-RL ayarları (`configs/mbrl.yaml`)

Derin RL kullanılmaz. Ajan doğrusal geçiş modelini veriden öğrenir, CEM ile planlar (`src/furnace_rl/agents/mbrl.py`).

| Hiperparametre | Değer |
|---|---|
| Model | Ridge regresyon, λ = 1e-3 |
| Yük özelliği | Bölge yükü + taze yük süzgeci, ρ = 0.995 |
| Planlama ufku | 60 adım (10 dk), eylem 60 s'lik parçalarla |
| Yeniden planlama | 60 s |
| CEM | nüfus 200, seçkin %10, 4 tur, σ = 0.5 |
| Bozucu kestirimi | süzgeç 0.05 |
| γ | 0.99 |
| Keşif | ilk 10 bölüm rastgele (σ = 0.3), sonra plan + σ = 0.1 |
| Toplam | 8 tur, 45 bölüm, 32 400 adım |
| Değerlendirme | her tur 5 sabit senaryo, en iyi model kaydedilir |
| Seed | 0, 1, 2 |

Karşılaştırma yöntemleri (3 seed): `configs/q_tile.yaml`, `configs/lspi.yaml`, `configs/ars.yaml`.

## 4. Deney listesi

| ID | Deney | Amaç | Sprint |
|---|---|---|---|
| E0 | Simülatör kalibrasyonu: sabit güç, rastgele politika, PID | Ortamın gerçekçi ve öğrenilebilir olduğunu göstermek | 1 |
| E1 | PID ızgara araması | Adil baseline | 1 |
| E2 | MB-RL – 3 seed eğitim | Öğrenme eğrisi (ortalama ± std), kararlılık | 2 |
| E3 | PID ve MB-RL – nominal test seti (20 senaryo) | Ana kıyas tablosu ve grafikleri | 2 |
| – | Yöntem kıyası: MB-RL, ARS, Q-öğrenme, LSPI | Ana yöntemin seçimini gerekçelendirmek | 2 |
| E4 | Ödül ağırlığı taraması `w₂ ∈ {0, 0.1, 0.2, 0.5, 1}`, `w₃ = 0` | Reward shaping yan etkileri, RMSE–kWh | 3 |
| E5 | Hiperparametre analizi (ufuk, yeniden planlama, CEM nüfusu, bozucu kestirimi, ρ, λ), 3 seed | Duyarlılık | 2–3 |
| E6 | Sensör gürültüsü σ ∈ {0, 1, 3, 5} °C | Gürbüzlük (rubrikte açıkça isteniyor) | 3 |
| E7 | Dağılım dışı senaryolar: 2500 kg parti, 0.05 / 0.4 m/dk bant, 8 dk arayla iki parti | Genelleme | 3 |
| E8 | Önizleme ablasyonu: gözlemde `m_next, v_next, t_arrival` yok | Önizleme bilgisinin katkısı | 3 |

## 5. Grafikler (rapor için)

1. Öğrenme eğrisi: adım – bölüm ödülü, 3 seed ortalaması ve gölgeli std bandı
2. Örnek senaryo zaman serisi: 3 bölge sıcaklığı (PID ve MB-RL üst üste, hedef kesikli çizgi),
   parti girişleri dikey çizgiyle işaretli
3. Aynı senaryoda güç komutları `u_i(t)`, PID ve MB-RL
4. Kıyas tablosu ve çubuk grafik: RMSE, maks. düşüş, kWh
5. E4 Pareto: x = kWh, y = RMSE, her nokta bir `w₂`
6. E6 gürültü: x = σ_noise, y = RMSE (PID ve MB-RL)
7. E8 önizleme ablasyonu: çubuk grafik

## 6. Çıktı düzeni

```
results/
├── models/mbrl_seed{0,1,2}/best_model.npz
├── logs/          # eğitim ve değerlendirme çıktıları
├── eval/          # her deney için metrics.csv
└── figures/       # rapora girecek .png dosyaları
```
