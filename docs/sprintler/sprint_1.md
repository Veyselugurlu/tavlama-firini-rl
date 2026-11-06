# Sprint 1 – Fırın Simülatörü ve PID Baseline ✅

**Hedef:** Gerçekçi davranan, test edilmiş bir `gymnasium` fırın ortamı ve adil biçimde ayarlanmış
PID baseline. Sprint sonunda "PID termal çöküşü geç telafi ediyor" grafiği elimizde olmalı.

**Rubrik karşılığı:** Simülasyon ortamı %25 (tamamı), Matematiksel formülasyon %20 (taslak),
Baseline %25 (PID kısmı)

**Durum:** Tamamlandı (25.10.2026)

## Görevler

### 1.1 Proje iskeleti
- [x] `requirements.txt`, `pyproject.toml` (paket `pip install -e .` ile kurulu)
- [x] Sanal ortam `.venv` (Python 3.11)
- [x] `src/furnace_rl/` paket yapısı. Varsayılan ayarlar `src/furnace_rl/config.py` içinde
      (ayrı bir `env_default.yaml` yerine tek kaynak kodda, deney YAML'ları yalnızca farkları içerir)
- [x] `.gitignore`

### 1.2 Senaryo üretici – `envs/scenarios.py`
- [x] Poisson varışlı parti takvimi
- [x] Sabit 20 senaryoluk test seti → `configs/test_scenarios.json` (seed 10000–10019)
- [x] Özel senaryolar → `configs/special_scenarios.json`

### 1.3 Ortam – `envs/furnace_env.py`
- [x] Bölge dengesi, iletim, parti ısınması (fizik alt adımlarıyla)
- [x] Aktüatör gecikmesi (τ_a + ölü zaman) ve termokupl gecikmesi + gürültü
- [x] 30 boyutlu normalize gözlem, `obs_slices` ile isimli dilimler
- [x] Ödül, terimler `info["reward_terms"]` içinde ayrı ayrı
- [x] İhlal adım başına cezalı, `truncated` (720 adım)
- [x] `use_preview: false` bayrağı (E8), `reset(options={"T_init": ...})` (adım yanıtı testleri)

### 1.4 Testler – `tests/test_env.py` (11 test, hepsi geçiyor)
- [x] `check_env`, seed tekrarlanabilirliği, soğuma/ısınma yönü, yüksüz denge,
      parti girince düşüş, enerji tutarlılığı, önizleme bayrağı, metrikler

### 1.5 Kalibrasyon (E0) – `scripts/calibrate_env.py`
- [x] Tüm kabul ölçütleri sağlandı, sonuçlar ve alınan kararlar `03_simulator_tasarimi.md` §3.1'de

### 1.6 PID baseline – `baselines/pid.py`, `scripts/tune_pid.py`
- [x] Anti-windup, ölçüm üzerinden filtreli türev, bumpless başlangıç
- [x] `PIDPolicy.predict(obs)`: RL politikalarıyla aynı arayüz
- [x] Kararlılık şartlı ızgara araması (E1) → `configs/pid.yaml`: Kp 0.04, Ti 300 s, Td 60 s

### 1.7 Metrik ve değerlendirme altyapısı
- [x] `metrics.py`, `evaluate.py` (CLI), `plots.py`

## Sonuçlar: PID, 20 test senaryosu (`results/eval/e3_pid/metrics.csv`)

| Metrik | Ortalama ± std |
|---|---|
| Bölüm ödülü | −885.9 ± 730.2 |
| RMSE | 19.16 ± 9.10 °C |
| Maks. düşüş | 68.4 ± 30.2 °C |
| Aşma | 16.1 ± 5.3 °C |
| B1 toparlanma | 20.5 ± 6.3 dk |
| Enerji | 690 ± 87 kWh |
| Eylem titreşimi | 0.052 |
| Güvenlik ihlali | 0 |

Grafik: `results/figures/e0_calibration_1ton.png` (`scripts/calibrate_env.py`).

**Gözlem:** PID gücü parti girdikten ve sıcaklık düşmeye başladıktan sonra artırıyor. 150 s
aktüatör gecikmesi ve 30 s ölü zaman yüzünden tepki geç kalıyor. Art arda gelen partilerde B1
doyuma girip 660 °C'ye kadar düşüyor, toparlanınca da 10–15 °C aşma yapıyor. RL ajanının
önizleme bilgisiyle kapatabileceği açık tam burası.

## Sprint 1 bitti sayılır (Definition of Done)
- [x] `pytest` yeşil
- [x] PID test setinde koşuyor, `results/eval/e3_pid/metrics.csv` üretildi
- [x] Örnek senaryoda PID sıcaklık ve güç grafiği `results/figures/` altında
- [x] Kalibre edilmiş parametre tablosu dokümanda güncel
- [x] MDP formülasyonu `docs/02_mdp_tasarimi.md` içinde yazıldı

## Tekrar üretmek için
```bash
.venv/Scripts/python scripts/make_test_scenarios.py
.venv/Scripts/python scripts/tune_pid.py --workers 7
.venv/Scripts/python scripts/calibrate_env.py
.venv/Scripts/python -m furnace_rl.evaluate --policy pid --out results/eval/pid
.venv/Scripts/python -m pytest -q
```
