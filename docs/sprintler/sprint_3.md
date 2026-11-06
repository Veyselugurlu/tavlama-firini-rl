# Sprint 3 – Gürbüzlük, Ablasyon ve Ödül Analizi ✅

**Hedef:** Rubriğin istediği kararlılık ve gürbüzlük analizlerini tamamlamak, reward shaping tartışmasını deneyle desteklemek.

**Rubrik karşılığı:** Eğitim/kararlılık %30 (tamamı), Formülasyon %20 (reward shaping tartışması)

Tüm değerler MB-RL (3 seed) ve 20 test senaryosundandır. Hiperparametreler test sonuçlarına bakılarak değiştirilmedi.

### 3.1 Sensör gürültüsü (E6) – `scripts/e6_e7_robustness.py`

| σ [°C] | PID RMSE | MB-RL RMSE | PID ödül | MB-RL ödül |
|---|---|---|---|---|
| 0 | 18.84 | **13.47** | −864 | **−520** |
| 1 (eğitim) | 19.16 | **13.63** | −886 | **−522** |
| 3 | 19.68 | **14.58** | −917 | **−570** |
| 5 | 20.29 | **15.81** | −959 | **−632** |

MB-RL her düzeyde iyi, ama gürültüden etkileniyor: σ = 5'te RMSE 2.3 °C artıyor (PID 1.4 °C). Eylem titreşimi ise neredeyse
sabit (0.028–0.030), PID'de 0.006'dan 0.071'e çıkıyor.

### 3.2 Ödül ağırlıkları (E4) – `scripts/plot_e4.py`, `scripts/plot_e4_smooth.py`

| Politika | RMSE [°C] | Maks. düşüş [°C] | Enerji [kWh] |
|---|---|---|---|
| PID | 19.16 | 68.4 | 690.1 |
| w₂ = 0 | 13.85 ± 0.55 | 50.9 | 695.5 ± 1.2 |
| w₂ = 0.1 | 13.96 ± 0.23 | 51.0 | 695.8 ± 0.1 |
| w₂ = 0.2 (ana) | 13.63 ± 0.28 | 50.1 | 697.3 ± 0.3 |
| w₂ = 0.5 | 13.70 ± 0.30 | 50.5 | 697.8 ± 0.8 |
| w₂ = 1.0 | 13.93 ± 0.31 | 50.5 | 697.0 ± 0.5 |

Beklenen ödünleşim yok, farklar seed değişkenliği düzeyinde: ~355 kWh duvar kaybı ve ~335 kWh parti ısıtması fizikten geliyor.
`w₃ = 0` ile titreşim de artmadı (ort. |Δu| 0.028, ana model 0.027, PID 0.052).

### 3.3 Dağılım dışı senaryolar (E7)

| Senaryo | PID RMSE | MB-RL RMSE | PID düşüş | MB-RL düşüş | PID aşma | MB-RL aşma |
|---|---|---|---|---|---|---|
| Tek ağır parti (1500 kg) | 21.0 | **13.0** | 77.8 | **40.9** | **15.3** | 21.7 |
| 2500 kg parti | 58.6 | **47.9** | 181.9 | **144.5** | **15.1** | 33.8 |
| 8 dk arayla iki parti | 42.3 | **34.9** | 144.8 | **121.2** | 15.1 | **12.3** |
| Hızlı bant 0.4 m/dk | 10.0 | **8.4** | 38.8 | **36.5** | **16.5** | 22.0 |
| Yavaş bant 0.05 m/dk | 6.2 | **2.9** | 37.1 | **6.9** | **7.3** | 11.7 |

### 3.4 Önizleme ablasyonu (E8) – `scripts/compare_e3.py --env-config configs/env_no_preview.yaml`, `scripts/plot_e8.py`

| Metrik | PID | Önizlemesiz | Önizlemeli |
|---|---|---|---|
| Bölüm ödülü | −885.9 | −971.7 ± 38.8 | **−522.4 ± 18.0** |
| RMSE [°C] | 19.16 | 20.30 ± 0.43 | **13.63 ± 0.28** |
| Maks. düşüş [°C] | 68.4 | 71.6 ± 1.1 | **50.1 ± 0.9** |
| Varış öncesi B1 güç artışı | −0.12 | −0.10 | **+0.30** |
| PID'den iyi olduğu senaryo | – | 2/20 | **20/20** |

Önizleme kalkınca üstünlük tamamen kayboluyor. PID'e bu bilgi verilmediği için kıyasta MB-RL lehine bilgi farkı var.

### 3.5 Hiperparametre analizi (E5) – `scripts/plot_e5.py`

| Ayar | Test ödülü | RMSE [°C] | Aşma [°C] |
|---|---|---|---|
| Referans (H = 60, λ = 1e-3, nüfus 200, ρ = 0.995) | −522.4 ± 18.0 | 13.63 ± 0.28 | 18.8 |
| Ufuk H = 30 | −617.5 ± 17.0 | 15.00 ± 0.27 | 14.9 |
| Ufuk H = 120 | −542.4 ± 16.8 | 14.33 ± 0.34 | 26.2 |
| Yeniden planlama 30 s | −521.5 ± 16.7 | 13.55 ± 0.25 | 17.5 |
| CEM nüfusu 50 | −532.3 ± 16.1 | 13.80 ± 0.25 | 19.4 |
| Bozucu kestirimi yok | −526.0 ± 10.4 | 13.63 ± 0.13 | 19.1 |
| Taze yük ρ = 0.99 | −548.6 ± 49.9 | 14.04 ± 0.83 | 19.6 |
| Taze yük ρ = 0.999 | −539.8 ± 21.3 | 14.00 ± 0.36 | 18.4 |
| Ridge λ = 1 | −558.3 ± 9.1 | 14.22 ± 0.12 | 18.6 |

### 3.6 Ödül yan etkileri – `scripts/reward_analysis.py`
- **Erken bitiş açığı:** tam güç kuralı, ihlalde biten bir tasarımda bölümü 97. adımda bitirip −700 alıyor; PID −886.
  Ortamda bu yüzden ihlal bölümü bitirmez.
- **Isı deposu:** MB-RL'de görülmedi. B1 yüksüz dönemde hedefin ~1 °C altında.

## Sprint 3 bitti sayılır
- [x] Gürültü, dağılım dışı, ablasyon, ödül ve hiperparametre deneyleri tamam
- [x] Tablolar `results/eval/`, şekiller `results/figures/` altında
