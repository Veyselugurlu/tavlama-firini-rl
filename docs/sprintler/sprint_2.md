# Sprint 2 – Q-Tablosu Ajanı ve PID Karşılaştırması ✅

**Hedef:** Derste işlenen tablolu Q-öğrenmeyi fırına uygulamak, 3 seed ile eğitmek ve aynı test senaryolarında PID ile kıyaslamak.

**Rubrik karşılığı:** Baseline kıyası %25 (tamamı), Eğitim ve hiperparametre %30 (kısmen)

**Kısıt:** Hoca şartı: yalnızca pekiştirmeli öğrenme, derin pekiştirmeli öğrenme yok. Ödev metninin önerdiği TD3 ve
PETS sinir ağı kullandığı için seçilmedi. Derste işlenen yöntem tablolu Q-öğrenme olduğu için ana yöntem odur.

## Görevler

### 2.1 Q-tablosu ajanı – `src/furnace_rl/agents/q_table.py`
- [x] Bölge başına ayrı Q-tablosu, ε-greedy, slayttaki güncelleme kuralı
- [x] Eylem: güç seviyesi seçimi; seviye ortamın artımlı eylemine çevrilir (±%10/adım sınırı korunur)
- [x] Karar periyodu: seviye k adım tutulur, k adımın indirgenmiş ödülü toplanır, güncellemede γᵏ kullanılır
- [x] Ortam ödülü bölgelere paylaştırılır (paylar toplamı ortam ödülüne eşit, `tests/test_agents.py`)
- [x] Ziyarete göre azalan öğrenme oranı: α(s,a) = max(α_min, α / (1 + n(s,a)/n₀))

### 2.2 Durum ızgarasının seçimi (yalnızca doğrulama senaryoları, seed 0)

| Durum temsili | Hücre (bölge başına) | Doğrulama RMSE [°C] |
|---|---|---|
| 11 hata × 5 hız × 4 yük × 4 yaklaşan yük, 11 güç seviyesi | 9 680 | 73.5 |
| 14 hata × 5 × 4 × 4, ziyarete göre azalan α | 12 320 | 34.7 |
| 14 hata × 5 × 3 × 3, 10 000 bölüm | 6 930 | 24.8 |
| **10 hata × 10 yük durumu, 6 güç seviyesi** | **600** | **13.3** |
| 10 hata × 10 değişim hızı (yük ve önizleme yok) | 600 | 22.3 |
| PID | – | 11.9 |

- Büyük tablolarda hücrelerin çoğu az güncellendi. Soğuk ama yüklü/yüksüz durumların aynı hücreye düşmesi (aliasing)
  ajanı düşük güç seçmeye itti.
- Hoca ızgaranın 10'luk olmasının şart olmadığını, daha küçük de olabileceğini söyledi; 10×10 seçildi.
- Eğitimde izleme cezasını sınırlamak (err_clip) denendi, sonucu kötüleştirdi, kullanılmadı.

### 2.3 Eğitim (E2)
- [x] `scripts/run_experiments.sh` tüm eğitimleri paralel çalıştırır (bitmiş koşuları atlar)
- [x] 3 seed, 5000 bölüm, 3.6 milyon adım; bir seed tek çekirdekte 45–55 dk
- [x] Referans ayar (`configs/q_table.yaml`) üzerinde E4, E5, E8 taramaları (Sprint 3)
- [x] Son ayar (`configs/q_table_son.yaml`, w₂ = 1): taramanın en iyi iki bulgusu + 30 s karar; doğrulama RMSE'sine göre seçildi

Doğrulama (5 senaryo, son ayar): en iyi RMSE 12.86–15.08 °C (ortalama 13.75). PID: 11.93 °C.

### 2.4 PID ile kıyas (E3) – `scripts/compare_e3.py`

| Metrik | PID | Q-tablosu (3 seed ort. ± std) | Fark |
|---|---|---|---|
| Bölüm ödülü | **−885.9 ± 730.2** | −1134.2 ± 92.0 | %28.0 daha kötü |
| RMSE | **19.16 ± 9.10 °C** | 22.37 ± 0.96 °C | %16.7 daha kötü |
| Maks. düşüş | **68.4 ± 30.2 °C** | 70.9 ± 3.7 °C | %3.5 daha kötü |
| Aşma | **16.1 ± 5.3 °C** | 27.6 ± 4.4 °C | %71 daha kötü |
| Enerji | 690.1 kWh | 686.3 kWh | %0.5 daha az |
| Eylem titreşimi | 0.052 | **0.028** | %46 daha az |
| 950 °C ihlali | 0 | 0 | |

Q-tablosu 20 senaryonun 3'ünde PID'den düşük RMSE, 7'sinde daha küçük maksimum düşüş aldı. Test sonucu
PID'den kötü çıktı ve olduğu gibi raporlandı; ayar test sonuçlarına bakılarak değiştirilmedi.

## Sprint 2 bitti sayılır
- [x] 3 seed eğitim tamam, öğrenme eğrileri çizildi
- [x] Kıyas tablosu ve grafikler `results/` altında
- [x] Kodda sinir ağı kütüphanesi yok (`tests/test_agents.py` denetler)
