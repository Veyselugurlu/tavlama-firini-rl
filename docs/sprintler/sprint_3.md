# Sprint 3 – Gürbüzlük, Ablasyon ve Ödül Analizi ✅

**Hedef:** Rubriğin istediği kararlılık ve gürbüzlük analizlerini tamamlamak, reward shaping tartışmasını deneyle desteklemek.

**Rubrik karşılığı:** Eğitim/kararlılık %30 (tamamı), Formülasyon %20 (reward shaping tartışması)

E6 ve E7 son ayarla (`q_son_seed*`), E4, E5 ve E8 referans ayarla (`q_table_seed*`) yapıldı. Tüm değerler 3 seed ve
20 test senaryosundandır. Ayarlar test sonuçlarına bakılarak değiştirilmedi.

### 3.1 Sensör gürültüsü (E6) – `scripts/e6_e7_robustness.py`

| σ [°C] | PID RMSE | Q-tablosu RMSE | PID ödül | Q-tablosu ödül |
|---|---|---|---|---|
| 0 | **18.84** | 21.46 ± 0.80 | **−864** | −1035 |
| 1 (eğitim) | **19.16** | 22.37 ± 0.96 | **−886** | −1134 |
| 3 | **19.68** | 24.24 ± 1.25 | **−917** | −1312 |
| 5 | **20.29** | 25.99 ± 2.24 | **−959** | −1512 |

Q-tablosu gürültüden PID'den çok etkileniyor: σ = 0'dan 5'e RMSE 4.5 °C artıyor (PID 1.4 °C). Hedefe yakın kutular
5–6 °C genişliğinde; gürültü ölçümü komşu kutuya atıp seçilen seviyeyi değiştiriyor.

### 3.2 Ödül ağırlıkları (E4) – `scripts/plot_e4.py`, `scripts/plot_e4_smooth.py`

| Politika | RMSE [°C] | Maks. düşüş [°C] | Enerji [kWh] |
|---|---|---|---|
| PID | 19.16 | 68.4 | 690.1 |
| w₂ = 0 | 30.12 ± 1.04 | 94.3 | 673.7 ± 0.9 |
| w₂ = 0.2 (referans) | 24.08 ± 4.01 | 74.1 | 686.3 ± 9.2 |
| w₂ = 1.0 | 22.12 ± 0.42 | 70.6 | 686.5 ± 3.4 |

- Enerji %2'lik aralıkta; tüketimi duvar kaybı ve parti ısıtması belirliyor.
- Beklenmeyen: w₂ büyüdükçe RMSE iyileşti. w₂ = 0'da ajan ısıtıcıları gereğinden az açtı. Bu bulgu son ayarda w₂ = 1'e yol açtı.
- w₃ = 0: RMSE 23.60, titreşim değişmedi (ort. |Δu| 0.020, yön değişimi oranı 0.004; PID 0.052 ve 0.415).

### 3.3 Dağılım dışı senaryolar (E7)

| Senaryo | PID RMSE | Q RMSE | PID düşüş | Q düşüş | PID aşma | Q aşma |
|---|---|---|---|---|---|---|
| Tek ağır parti (1500 kg) | 21.0 | **19.2** | 77.8 | **63.3** | **15.3** | 19.1 |
| 2500 kg parti | 58.6 | **57.5** | 181.9 | **162.6** | **15.1** | 33.4 |
| 8 dk arayla iki parti | **42.3** | 43.5 | 144.8 | **139.6** | **15.1** | 22.9 |
| Hızlı bant 0.4 m/dk | **10.0** | 11.4 | **38.8** | 42.5 | **16.5** | 18.5 |
| Yavaş bant 0.05 m/dk | **6.2** | 18.3 | 37.1 | **30.4** | **7.3** | 59.8 |

Tek büyük yükte Q-tablosu iyi (hemen tam güç), uzun süre hedefte durulması gereken yavaş bantta kötü (iki seviye
arasında salınım).

### 3.4 Önizleme ablasyonu (E8) – `scripts/compare_e3.py --env-config configs/env_no_preview.yaml`, `scripts/plot_e8.py`

| Metrik | PID | Önizlemesiz | Önizlemeli (referans) |
|---|---|---|---|
| RMSE [°C] | **19.16** | 24.04 ± 1.01 | 24.08 ± 4.01 |
| Maks. düşüş [°C] | **68.4** | 73.6 | 74.1 |
| Aşma [°C] | **16.1** | 27.6 | 31.9 |
| Varış öncesi B1 güç artışı | −0.12 | −0.13 | −0.04 |

Önizleme testte katkı yapmadı. "Parti yaklaşıyor" hücreleri seyrek görülüyor ve önceden ısıtmanın faydası gecikmeli geliyor.

### 3.5 Hiperparametre analizi (E5) – `scripts/plot_e5.py`

| Ayar | Test ödülü | RMSE [°C] | Aşma [°C] |
|---|---|---|---|
| Referans (60 s, seviyeler 0/0.15/0.25/0.35/0.5/1, w₂ = 0.2) | −1320.9 ± 429.6 | 24.08 ± 4.01 | 31.9 |
| Güç seviyeleri 0/0.2/0.3/0.4/0.6/1 | −1120.7 ± 125.3 | 22.56 ± 1.59 | 29.9 |
| α_min = 0.05 | −1423.3 ± 257.1 | 25.30 ± 2.38 | 31.4 |
| ε 0.5 → 0.05 | −1485.8 ± 220.8 | 26.27 ± 2.03 | 33.6 |
| Izgara: hata × değişim hızı (yük yok) | −2351.7 ± 575.3 | 33.00 ± 3.81 | 32.9 |
| Seviyeler + w₂ = 1 | −1291.7 ± 142.9 | 23.90 ± 1.48 | 24.6 |
| **Son ayar:** seviyeler + w₂ = 1 + 30 s karar | **−1134.2 ± 92.0** | 22.37 ± 0.96 | 27.6 |

Son ayar doğrulama RMSE'sine göre seçildi (referans 14.31, seviyeler + w₂ = 1: 14.15, son ayar 13.75 °C), test sonra çalıştırıldı.

### 3.6 Ödül yan etkileri – `scripts/reward_analysis.py`
- **Erken bitiş açığı:** tam güç kuralı, ihlalde biten bir tasarımda bölümü 97. adımda bitirip −700 alıyor; PID −886.
  Ortamda bu yüzden ihlal bölümü bitirmez.
- **Isı deposu:** görülmedi. B1 varıştan önceki 10 dakikada hedefin 10.2–13.8 °C altında (PID 11.8 °C).

## Sprint 3 bitti sayılır
- [x] Gürültü, dağılım dışı, ablasyon, ödül ve hiperparametre deneyleri tamam
- [x] Tablolar `results/eval/`, şekiller `results/figures/` altında
