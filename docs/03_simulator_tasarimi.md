# 03 – Simülatör Tasarımı

Hedef: 3B ya da CFD olmadan, birkaç fark denklemiyle fırının üç davranışını yakalamak:
1. **Yüksek ısıl eylemsizlik:** bölge sıcaklığı yavaş değişir.
2. **Gecikme:** komut verilen güç hemen etki etmez, termokupl da geç gösterir.
3. **Termal çöküş:** soğuk parti girince bölge sıcaklığı belirgin düşer.

## 1. Fiziksel yapı

```
 parti girişi →  [ Bölge 1: ön ısıtma ] → [ Bölge 2: ısıtma ] → [ Bölge 3: tutma ] → çıkış
                     T₁, u₁                   T₂, u₂                 T₃, u₃
          ← kayıp →        ← iletim k →             ← iletim k →          ← kayıp →
```

- Her bölge tek bir ısıl kütle (lumped capacitance): refrakter + iç atmosfer
- Komşu bölgeler birbiriyle ısı alışverişi yapar
- Partiler bant üzerinde hızla `v` ilerler; bir parti her an tam bir bölgededir (nokta kütle)

## 2. Denklemler (Euler, `Δt = 10 s`)

**Aktüatör (eylemsizlik + ölü zaman):**
```
u_cmd_i(t)   = eylem, d adım gecikmeyle uygulanır  → u_del_i(t) = u_cmd_i(t − d)
u_eff_i(t+1) = u_eff_i(t) + (Δt / τ_a) · (u_del_i(t) − u_eff_i(t))
```

**Bölge enerji dengesi:**
```
C · (T_i(t+1) − T_i(t)) / Δt =
      P_max · u_eff_i                         ← ısıtma
    − h · (T_i − T_amb)                        ← duvardan kayıp
    − k · (T_i − T_{i−1}) − k · (T_i − T_{i+1})  ← komşu bölgeler (sınır bölgelerde tek taraf)
    − Σ_{j ∈ bölge i} Q_j                      ← o bölgedeki partilere giden ısı
```

**Parti ısınması** (j. parti, kütle `m_j`, sıcaklık `T_m,j`):
```
Q_j          = hA_m · m_j · (T_i − T_m,j)
m_j · c_m · (T_m,j(t+1) − T_m,j(t)) / Δt = Q_j
x_j(t+1)     = x_j(t) + v_j · Δt               ← konum; bölge = floor(x / L)
```

**Termokupl (ölçüm gecikmesi + gürültü):**
```
T_tc_i(t+1)  = T_tc_i(t) + (Δt / τ_tc) · (T_i(t) − T_tc_i(t))
T_meas_i     = T_tc_i + N(0, σ_noise²)
```

**Enerji:**
```
E_t [kWh] = Σ_i P_max · u_eff_i · Δt / 3.6e6
```

> Euler kararlılığı: `Δt / τ_a < 1` ve `Δt / τ_tc < 1` olmalı (seçilen değerlerle 0.17 ve 0.5).
> Gerekirse fizik 1 s'lik alt adımlarla hesaplanır, kontrol 10 s'de bir verilir.

## 3. Parametreler (Sprint 1'de kalibre edildi)

Kodda tek kaynak: `src/furnace_rl/config.py` → `DEFAULT_ENV_CONFIG`. Deney YAML'ları yalnızca
değişen anahtarları içerir.

| Parametre | Sembol | Değer | Not |
|---|---|---|---|
| Bölge sayısı | N | 3 | config ile değişebilir |
| Kontrol adımı | Δt | 10 s | fizik 2 s'lik 5 alt adımla |
| Bölüm süresi | — | 720 adım = 2 saat | |
| Hedef profil | T_ref | [750, 850, 850] °C | ön ısıtma / ısıtma / tutma |
| Ortam sıcaklığı | T_amb | 25 °C (bölüm başında ±10) | |
| Güvenlik üst sınırı | T_max | 950 °C | |
| Bölge maksimum gücü | P_max | [250, 200, 160] kW | giriş bölgesi en güçlü |
| Bölge ısıl kapasitesi | C | 1.0 MJ/K | zaman sabiti C/h ≈ 3.7 saat |
| Duvar kayıp katsayısı | h | 75 W/K | 850 °C'de ≈ 62 kW kayıp |
| Bölgeler arası iletim | k | 30 W/K | |
| Aktüatör zaman sabiti | τ_a | 150 s | ısıl eylemsizliğin ana kaynağı |
| Aktüatör ölü zamanı | d | 3 adım (30 s) | |
| Termokupl zaman sabiti | τ_tc | 20 s | |
| Sensör gürültüsü | σ_noise | 1 °C (eğitim), 0–5 °C (test) | |
| Bölge uzunluğu | L | 5 m | |
| Parti kütlesi | m | U(300, 1500) kg | test: 2500 kg (dağılım dışı) |
| Parti hızı | v | U(0.1, 0.3) m/dk | bölgede 17–50 dk kalış |
| Parti giriş sıcaklığı | T_m,0 | 25 °C | |
| Malzeme özgül ısısı | c_m | 0.6 kJ/(kg·K) | çelik, ortalama |
| Malzeme ısı transferi | hA_m | 0.3 W/(K·kg) | 1 tonda 300 W/K; ısınma zaman sabiti ≈ 33 dk |
| Parti varışı | — | Poisson, ort. 35 dk, en az 15 dk aralık | 2 saatte 2–7 (çoğunlukla 3–5) parti |
| Önizleme ufku | t_preview | 10 dk | parti bu süre kala "duyurulur" |

**Basitleştirmeler (raporda belirtilecek):** her parti nokta kütle ve kendi hızıyla ilerler
(sollama kısıtı yok); bölge içi sıcaklık homojen; radyasyon doğrusal ısı transferiyle yaklaşıklandı.

## 3.1 Kalibrasyon sonuçları (E0, `scripts/calibrate_env.py`)

| Ölçüt | Hedef | Sonuç |
|---|---|---|
| Yüksüz denge gücü B1/B2/B3 | %20–60 | %20.5 / %32.4 / %38.7 (hedefte) |
| Tam güçte ısınma hızı | 5–20 °C/dk | 11.0 / 7.6 / 5.5 °C/dk (hedefte) |
| 1 ton parti, sabit güç (kontrolsüz) B1 düşüşü | — | 175 °C (problemin büyüklüğü) |
| 1 ton parti, ayarlı PID, B1 düşüşü | 30–80 °C | 33.8 °C (hedefte) |
| 1 ton parti, ayarlı PID, B1 toparlanma (±5 °C) | — | 34 dk (ısıl eylemsizlik nedeniyle uzun) |
| Rastgele takvimde PID doyumu B1/B2/B3 | ara sıra | %37 / %22 / %9 (hedefte) |

**Kalibrasyon sırasında alınan kararlar:**
1. İlk değerlerle (P_max 350 kW, τ_a 60 s) PID düşüşü yalnızca 17 °C idi ve PID neredeyse
   hiç doymuyordu, yani problem RL'e yer bırakmayacak kadar kolaydı.
2. Güç azaltılınca (P_max 200 kW) fırın kapasitesi yetersiz kaldı. Bu durumda en iyi ve en kötü
   PID kazançları neredeyse aynı ödülü verdi, yani hiçbir kontrolcü fark yaratamıyordu.
3. Son seçim: kapasite yeterli (250 kW), ama ısıl eylemsizlik yüksek (τ_a = 150 s, ölü zaman
   30 s). Bu, ödevin tanımladığı probleme de uygun: PID geç tepki veriyor, partiyi önceden gören
   ajanın telafi edebileceği bir açık kalıyor.
4. Varış aralığı en az 8 dk iken bazı senaryolar 2 saatte 6–7 ton yüklüyordu ve fırın 237 °C
   çöküyordu (gerçekçi olmayan üretim planı). En az aralık 15 dk, ortalama 35 dk yapıldı.

## 4. Senaryo üretici (`scenarios.py`)

- `generate_schedule(rng, duration, cfg) → [(t_arrival, mass, speed), ...]`
- **Eğitim:** her `reset`'te rastgele takvim (domain randomization: kütle, hız, aralık, `T_amb`)
- **Test seti:** sabit seed'lerle üretilmiş 20 senaryo, `configs/test_scenarios.yaml` ya da
  `.json` olarak kaydedilir. PID ve RL ajanları bu senaryolarda birebir aynı koşullarda çalışır.
- **Özel senaryolar** (`configs/special_scenarios.json`): tek ağır parti (1500 kg), art arda
  yakın iki parti (8 dk arayla, eğitim dağılımı dışında), dağılım dışı ağır parti (2500 kg),
  yavaş bant (0.05 m/dk), hızlı bant (0.4 m/dk)

## 5. `gymnasium.Env` arayüzü

```python
class FurnaceEnv(gym.Env):
    def __init__(self, config: dict | None = None, scenario: list | None = None): ...
    def reset(self, seed=None, options=None) -> tuple[obs, info]: ...
    def step(self, action) -> tuple[obs, reward, terminated, truncated, info]: ...
```

`info` içeriği: `T_true`, `T_meas`, `u_eff`, `energy_kwh`, `reward_terms` (her ödül teriminin
ayrı değeri), `batches` (konum/sıcaklık). Değerlendirme ve grafikler bu alanlardan beslenir.

## 6. Doğrulama testleri (`tests/test_env.py`)

- `gymnasium.utils.env_checker.check_env` geçmeli
- Aynı seed ile iki koşu birebir aynı gözlemleri üretmeli
- `u = 0` iken bölgeler `T_amb`'e doğru soğumalı, `u = 1` iken ısınmalı
- Parti girince bulunduğu bölgenin sıcaklığı yüksüz duruma göre düşmeli
- Enerji her adımda ≥ 0, toplamı `Σ E_t`'ye eşit olmalı
