# Çok Bölgeli Tavlama Fırınında Pekiştirmeli Öğrenme ile Sıcaklık ve Enerji Optimizasyonu

Yüksek lisans Pekiştirmeli Öğrenme dersi ödevi.

**Problem:** Sürekli çalışan bir metal ısıl işlem fırınına soğuk malzeme partisi girdiğinde bölge
sıcaklıkları çöker. Fırının ısıl eylemsizliği yüksek olduğu için geri beslemeli bir kontrolcü bu
çöküşü ancak hata oluştuktan sonra telafi edebilir.

**Yöntem:** Kontrol yalnızca pekiştirmeli öğrenmeyle yapılır ve **derin pekiştirmeli öğrenme kullanılmaz: projede sinir
ağı yoktur.** Ana yöntem model tabanlı RL'dir (MB-RL, ödevin önerdiği MPC-RL ailesi): ajan fırının geçiş modelini
etkileşim verisinden doğrusal regresyonla öğrenir, gelecek partinin kütlesini ve hızını önceden görerek çapraz entropi
yöntemiyle (CEM) 3 bölgenin ısıtıcı gücünü planlar. PID yalnızca karşılaştırma ölçütüdür (baseline); ajanın eğitimine
ya da kararlarına karışmaz.

## Değerlendirme ölçütleri nerede?

| Ölçüt | Ağırlık | Nerede |
|---|---|---|
| Matematiksel formülasyon (S, A, P, R) ve ödül şekillendirmenin yan etkileri | %20 | [docs/02_mdp_tasarimi.md](docs/02_mdp_tasarimi.md), [ödül yan etkileri](#ödül-tasarımının-yan-etkileri) |
| Simülasyon ortamı (`gymnasium.Env`, `reset`/`step`) | %25 | [src/furnace_rl/envs/furnace_env.py](src/furnace_rl/envs/furnace_env.py), [docs/03_simulator_tasarimi.md](docs/03_simulator_tasarimi.md) |
| Klasik yöntemle karşılaştırma (PID) | %25 | [src/furnace_rl/baselines/pid.py](src/furnace_rl/baselines/pid.py), [karşılaştırma tablosu](#pid-ile-karşılaştırma) |
| Eğitim, hiperparametre ve kararlılık (3 seed, sensör gürültüsü) | %30 | [src/furnace_rl/agents/mbrl.py](src/furnace_rl/agents/mbrl.py), [öğrenme eğrileri](#öğrenme-eğrileri), [gürültü testi](#sensör-gürültüsü) |

## Sonuçlar

Tüm sayılar eğitimde kullanılmayan 20 test senaryosundan gelir (seed 10000–10019). MB-RL: 3 seed × 32 400 adım
(tek çekirdekte seed başına 6–7 dk). Hiperparametreler test sonuçlarına bakılarak değiştirilmedi.

### PID ile karşılaştırma

| Metrik | PID | MB-RL (3 seed ort. ± std) | Fark |
|---|---|---|---|
| Bölüm ödülü | −885.9 ± 730.2 | **−522.4 ± 18.0** | %41.0 daha iyi |
| RMSE | 19.16 ± 9.10 °C | **13.63 ± 0.28 °C** | %28.9 daha iyi |
| En büyük sıcaklık düşüşü | 68.4 ± 30.2 °C | **50.1 ± 0.9 °C** | %26.7 daha iyi |
| Aşma | **16.1 ± 5.3 °C** | 18.8 ± 0.5 °C | daha kötü |
| B1 toparlanma (±5 °C) | **20.5 dk** | 25.2 dk | daha kötü |
| Enerji | **690 kWh** | 697 kWh | %1.0 daha fazla |
| 950 °C ihlali | 0 dk | 0 dk | |

PID sütunundaki std senaryolar arası, MB-RL sütunundaki std seed'ler arasıdır. MB-RL, 20 senaryonun tamamında PID'den
yüksek ödül aldı.

![PID ve MB-RL çubuk grafiği](results/figures/e3_bars.png)

![test_05 senaryosunda PID ve MB-RL](results/figures/e3_test_05.png)

**Avantajın kaynağı önizleme bilgisi.** Gelecek parti gözlemden çıkarılıp ajan aynı ayarlarla yeniden eğitilince üstünlük
tamamen kayboluyor:

| Metrik | PID | MB-RL önizlemesiz | MB-RL önizlemeli |
|---|---|---|---|
| Bölüm ödülü | −885.9 | −971.7 ± 38.8 | **−522.4 ± 18.0** |
| RMSE | 19.16 °C | 20.30 ± 0.43 °C | **13.63 ± 0.28 °C** |
| En büyük düşüş | 68.4 °C | 71.6 ± 1.1 °C | **50.1 ± 0.9 °C** |
| Varış öncesi B1 güç artışı | −0.12 | −0.10 | **+0.30** |
| PID'den iyi olduğu senaryo | – | 2/20 | **20/20** |

Bu nedenle kıyasta MB-RL lehine bir bilgi farkı vardır: parti bilgisini kullanan ileri beslemeli bir PID aradaki farkı
küçültebilir.

### Yöntem karşılaştırması

Sinir ağı içermeyen diğer yöntemler de aynı ortamda 3 seed ile denendi:

| Yöntem | Eğitim adımı | Test ödülü | RMSE [°C] | En büyük düşüş [°C] |
|---|---|---|---|---|
| PID (baseline) | – | −885.9 | 19.16 | 68.4 |
| **MB-RL** (doğrusal model + CEM) | **32 400** | −522.4 | 13.63 | 50.1 |
| ARS (doğrusal politika) | 11 520 000 | −466.8 | 12.66 | 45.8 |
| Q-öğrenme (tile coding) | 2 160 000 | −8209.8 | 62.43 | 155.8 |
| LSPI (karesel Q) | 118 800 | −9027.0 | 66.29 | 165.7 |

ARS testte MB-RL'den biraz iyi, ama yaklaşık 355 kat fazla etkileşim kullanıyor ve bir rastgele arama yöntemi olduğu için
ana yöntem olarak seçilmedi. Q-öğrenme ve LSPI PID'in çok gerisinde kaldı.

![Yöntemlerin öğrenme eğrileri](results/figures/methods_learning.png)

### Öğrenme eğrileri

Doğrulama seti (5 senaryo, seed 5000–5004), 3 seed ortalaması ± std. İlk model (10 rastgele bölüm, 7200 adım) PID'i geçiyor,
ikinci turdan sonra eğri yataylaşıyor. En iyi modeller: ödül −165 ile −172 arası (PID −364), RMSE 6.78–7.00 °C (PID 11.93 °C).

![Öğrenme eğrisi](results/figures/e2_return.png)

Hiperparametre analizi (her ayar 3 seed, [results/eval/e5_hparams.md](results/eval/e5_hparams.md)): en belirgin etki
planlama ufkunda. 5 dakikalık ufuk RMSE'yi 15.00 °C'ye çıkarıyor, 20 dakikalık ufuk aşmayı 26 °C'ye çıkarıyor (referans: 10 dk,
13.63 °C, 18.8 °C). Daha sık yeniden planlama ve bozucu kestirimi seed değişkenliği içinde kalan farklar üretiyor.

### Sensör gürültüsü

Yeniden eğitim yapılmadan, 20 test senaryosunda (eğitim gürültüsü σ = 1 °C):

| σ [°C] | PID RMSE | MB-RL RMSE | PID ödül | MB-RL ödül |
|---|---|---|---|---|
| 0 | 18.84 | **13.47** | −864 | **−520** |
| 1 | 19.16 | **13.63** | −886 | **−522** |
| 3 | 19.68 | **14.58** | −917 | **−570** |
| 5 | 20.29 | **15.81** | −959 | **−632** |

MB-RL her düzeyde PID'den iyi, ama gürültüden etkileniyor (σ = 5'te +2.3 °C, PID'de +1.4 °C).

![Gürültü testi](results/figures/e6_noise.png)

### Eğitimde görülmemiş senaryolar

| Senaryo | PID RMSE | MB-RL RMSE | PID düşüş | MB-RL düşüş |
|---|---|---|---|---|
| Tek ağır parti (1500 kg) | 21.0 | **13.0** | 77.8 | **40.9** |
| 2500 kg parti (eğitimde en çok 1500) | 58.6 | **47.9** | 181.9 | **144.5** |
| 8 dk arayla iki parti (eğitimde en az 15) | 42.3 | **34.9** | 144.8 | **121.2** |
| Hızlı bant 0.4 m/dk (eğitimde en çok 0.3) | 10.0 | **8.4** | 38.8 | **36.5** |
| Yavaş bant 0.05 m/dk (eğitimde en az 0.1) | 6.2 | **2.9** | 37.1 | **6.9** |

Aşma ise 5 senaryonun 4'ünde PID'den büyük (2500 kg partide 33.8 °C, PID 15.1 °C).

### Ödül tasarımının yan etkileri

- **Erken bitiş açığı:** 950 °C aşımı bölümü bitirseydi, ödül hep negatif olduğu için bölümü kısaltmak kârlı olurdu. Eğitimsiz
  ölçüldü: tam güç kuralı bölümü ortalama 97. adımda bitirip −700 alıyor, PID tüm bölümde −886 alıyor. Bu yüzden ihlal bölümü
  bitirmez, her ihlal adımı cezalandırılır.
- **Kare hatadan beklenen ısı deposu görülmedi:** B1 yüksüz dönemde hedefin ~1 °C altında. Ajan sürekli fazla sıcak durmak
  yerine gücü varıştan kısa süre önce artırıyor.
- **Enerji ağırlığı tüketimi değiştirmedi:** w₂ ∈ {0, 0.1, 0.2, 0.5, 1} için enerji 695.5–697.8 kWh aralığında kaldı. Bir bölümde
  ~355 kWh duvar kaybına, ~335 kWh parti ısıtmaya gidiyor; kontrolcünün etkileyebildiği pay küçük. 10 °C düşük durmanın enerji
  kazancı, aynı sapmanın izleme cezasının ~70'te biri.
- **Titreşim cezası (w₃ = 0):** Beklenen titreşim artışı görülmedi (ort. |Δu| 0.028, ana model 0.027).

![Enerji ağırlığı taraması](results/figures/e4_pareto.png)

**Sınırlılıklar:** Simülatör parametreleri gerçek bir fırından alınmadı, kalibre edilerek seçildi. PID ileri beslemesizdir.
Planlayıcı ödül fonksiyonunu, ölü zaman adım sayısını ve partinin giriş bölgesini bilir; ısıl parametreler ise veriden öğrenilir.
Taze yük süzgeci ρ, simülatördeki parti ısınma süresine göre seçildi. Yöntem, fırının yaklaşık doğrusal olmasından yararlanır.

## Kurulum

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install -e .
.venv/Scripts/python -m pytest -q      # 20 test
```

Komutlar Windows yollarıyla yazılmıştır. Konsolda Türkçe karakterler bozuk görünürse önce `set PYTHONIOENCODING=utf-8`.
Betikleri Git Bash ile çalıştırın. Sanal ortam Linux veya macOS'ta farklı bir yerdeyse `PY=.venv/bin/python` verin.

## Sonuçları yeniden üretme

Ana deneyin 3 modeli depoda (`results/models/mbrl_seed*/best_model.npz`), bu yüzden şu adımlar eğitim gerektirmez:

```bash
.venv/Scripts/python scripts/compare_e3.py --runs results/models/mbrl_seed0 results/models/mbrl_seed1 results/models/mbrl_seed2 --tag e3 --label MB-RL
.venv/Scripts/python scripts/e6_e7_robustness.py --runs results/models/mbrl_seed0 results/models/mbrl_seed1 results/models/mbrl_seed2
```

Tek bir eğitim:

```bash
.venv/Scripts/python -m furnace_rl.agents.mbrl --seed 0 --name mbrl_seed0
```

Tüm eğitimler ve değerlendirmeler (54 koşu, 7 paralel iş ile ~1.5 saat; bitmiş koşular atlanır):

```bash
bash scripts/run_experiments.sh 7
bash scripts/run_evaluations.sh 7
```

Tüm şekiller `results/figures/`, tablolar `results/eval/` altındadır.

## Klasör yapısı

```
├── configs/                   # ortam, PID, RL ayarları, test senaryoları
├── docs/                      # gereksinim, MDP, simülatör ve deney tasarımı, sprint kayıtları
├── results/
│   ├── eval/                  # sonuç tabloları (csv, md)
│   ├── figures/               # grafikler
│   └── models/mbrl_seed*/     # ana deneyin en iyi modelleri ve doğrulama kayıtları
├── scripts/                   # deneyleri çalıştıran betikler
├── src/furnace_rl/
│   ├── envs/                  # gymnasium ortamı, senaryo üretici
│   ├── agents/mbrl.py         # ana yöntem: doğrusal model + CEM planlama
│   ├── agents/                # karşılaştırma: q_tile.py, lspi.py, ars.py
│   ├── baselines/pid.py       # karşılaştırma için PID
│   ├── evaluate.py, metrics.py, plots.py
│   └── config.py              # ortam ayarlarının tek kaynağı
└── tests/                     # ortam, ajan ve metrik testleri
```

## Teknoloji

Python 3.11, `gymnasium`, `numpy`, `pandas`, `matplotlib`, `pyyaml`, `pytest`. Sinir ağı kütüphanesi kullanılmaz, GPU gerekmez.
`tests/test_agents.py` kodda ve bağımlılıklarda sinir ağı kütüphanesi olmadığını otomatik denetler.
