# Sprint 2 – RL Ajanı ve PID Karşılaştırması ✅

**Hedef:** Sinir ağı içermeyen RL ajanını 3 seed ile eğitmek ve aynı test senaryolarında PID ile kıyaslamak.

**Rubrik karşılığı:** Baseline kıyası %25 (tamamı), Eğitim ve hiperparametre %30 (kısmen)

**Kısıt:** Hoca şartı: yalnızca pekiştirmeli öğrenme, derin pekiştirmeli öğrenme yok. PID + RL hibriti kullanılmaz.
Ödev metninin önerdiği yöntemlerden TD3 ve PETS sinir ağı kullandığı için, önerinin model tabanlı RL / MPC-RL kolu
doğrusal modelle uygulandı.

## Görevler

### 2.1 Yöntem denemeleri (hepsi sinir ağısız, aynı doğrulama setinde)
- [x] LSPI, karesel Q (`agents/lspi.py`): kararsız. Q'nun eylem eğriliği yanlış işarete dönüyor, kazançlar dağılıyor.
- [x] Q-öğrenme, tile coding (`agents/q_tile.py`): öğreniyor ama PID düzeyine ulaşmıyor.
  İlk denemede sıfır başlangıç değeri, tüm gerçek Q'lar negatif olduğu için denenmemiş eylemleri en iyi gösteriyordu; düzeltildi.
- [x] ARS, doğrusal politika (`agents/ars.py`): PID'i 2.4–4.5 milyon adımda geçiyor. Rubrikte rastgele arama klasik yöntem
  sayıldığından ana yöntem olmadı.
- [x] MB-RL: doğrusal model + CEM (`agents/mbrl.py`): ilk modelden itibaren PID'i geçiyor. **Ana yöntem.**

### 2.2 Eğitim (E2)
- [x] `scripts/run_experiments.sh` tüm eğitimleri paralel çalıştırır
- [x] 3 seed, 8 tur, 32 400 adım, bir seed tek çekirdekte 6–7 dk

Doğrulama (5 senaryo): en iyi RMSE 6.78–7.00 °C, ödül −165 ile −172 arası. PID: 11.93 °C, −364.

### 2.3 PID ile kıyas (E3) – `scripts/compare_e3.py`

| Metrik | PID | MB-RL (3 seed ort. ± std) | Fark |
|---|---|---|---|
| Bölüm ödülü | −885.9 ± 730.2 | **−522.4 ± 18.0** | %41.0 daha iyi |
| RMSE | 19.16 ± 9.10 °C | **13.63 ± 0.28 °C** | %28.9 daha iyi |
| Maks. düşüş | 68.4 ± 30.2 °C | **50.1 ± 0.9 °C** | %26.7 daha iyi |
| Aşma | **16.1 ± 5.3 °C** | 18.8 ± 0.5 °C | daha kötü |
| B1 toparlanma | **20.5 dk** | 25.2 dk | daha kötü |
| Enerji | **690 kWh** | 697 kWh | %1.0 fazla |
| Varış öncesi B1 güç artışı | −0.12 | **+0.30** | |
| 950 °C ihlali | 0 dk | 0 dk | |

MB-RL 20 senaryonun tamamında PID'den yüksek ödül aldı.

### 2.4 Yöntem kıyası – `scripts/compare_methods.py`

| Yöntem | Eğitim adımı | Test ödülü | RMSE [°C] | Maks. düşüş [°C] |
|---|---|---|---|---|
| PID | – | −885.9 | 19.16 | 68.4 |
| MB-RL | 32 400 | −522.4 | 13.63 | 50.1 |
| ARS | 11 520 000 | −466.8 | 12.66 | 45.8 |
| Q-öğrenme, tile coding | 2 160 000 | −8209.8 | 62.43 | 155.8 |
| LSPI | 118 800 | −9027.0 | 66.29 | 165.7 |

## Sprint 2 bitti sayılır
- [x] 3 seed eğitim tamam, öğrenme eğrileri çizildi
- [x] Kıyas tablosu ve grafikler `results/` altında
- [x] Kodda sinir ağı kütüphanesi yok (`tests/test_agents.py` denetler)
