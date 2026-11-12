# 02 – MDP Tasarımı

Problem bir MDP `(S, A, P, R, γ)` olarak modellenir. Sensör gecikmesi ve gürültü yüzünden gerçekte
kısmen gözlenebilir (POMDP), ama türevler, önceki eylem ve önizleme bilgisi gözleme eklendiği için
MDP yaklaşımı yeterli kabul edilir. Bu varsayım raporda açıkça belirtilmeli.

- Zaman adımı: `Δt = 10 s`
- Bölüm uzunluğu: 720 adım (2 saat çalışma)
- İndirim faktörü: `γ = 0.99` (efektif ufuk ~100 adım ≈ 17 dk). Isıl gecikmeleri kapsamaya yetiyor.

## 1. Durum / Gözlem uzayı `S`

N = 3 bölge için gözlem vektörü (tümü normalize, yaklaşık `[-1, 1]`):

| Bileşen | Boyut | Açıklama | Normalizasyon |
|---|---|---|---|
| `T_meas_i` | N | Termokupl ölçümü (gecikmeli + gürültülü) | `(T − T_ref_i) / 25` |
| `dT_i` | N | Ölçülen sıcaklığın türevi (°C/dk) | `/ 10` |
| `T_ref_i` | N | Hedef profil (ileride değişken profil için) | `/ 1000` |
| `u_prev_i` | N | Bir önceki eylem | zaten `[0, 1]` |
| `m_next` | 1 | Gelecek partinin kütlesi (yoksa 0) | `/ m_max` |
| `v_next` | 1 | Gelecek partinin hızı | `/ v_max` |
| `t_arrival` | 1 | Gelecek partinin varışına kalan süre (önizleme ufku içinde) | `/ t_preview` |
| `load_i` | N | Her bölgede şu an bulunan toplam malzeme kütlesi (izleme sistemi) | `/ m_max` |
| `u_eff_i` | N | Isıtıcının fiilen verdiği güç oranı (güç ölçeri; kWh zaten ölçülüyor) | zaten `[0, 1]` |
| `pending` | d·N | Ölü zamanda bekleyen son d = 3 komut (kontrolcü kendi geçmişini bilir) | zaten `[0, 1]` |

Toplam boyut: `6N + 3 + dN = 30` (N = 3, d = 3 için).

Ölçekler: sıcaklık hatası `/ 25` (±250 °C'de kırpılır), türev `/ 10` °C/dk.

> Ödev metnindeki "sıcaklıklar, türevler, gelecek malzemenin kütlesi ve hızı" doğrudan karşılanıyor.
> `u_prev`, `load_i` ve `t_arrival` Markov özelliğini güçlendirmek için eklendi. Bu ekleme
> raporda gerekçelendirilmeli.

### 1.1 Neden aktüatör durumu gözlemde?

Bir güç komutu 30 s ölü zamandan sonra ve 150 s zaman sabitiyle ısıtıcıya ulaşıyor. Gözlemde
yalnızca son komut (`u_prev`) varken ajan, önceki komutların ne kadarının "yolda" olduğunu
bilemiyor; aynı gözlem farklı gelecekler doğurabiliyor (Markov özelliği bozuk). Gecikmeli MDP'lerde
standart çözüm durumu bekleyen eylemlerle genişletmektir. Ayrıca fiili ısıtıcı gücü sahada güç
ölçeriyle ölçülür, bu yüzden gözleme eklenmesi gerçekçidir.

**Q-tablosunun gördüğü durum:** Q-tablosu bu vektörün tamamını kullanmaz. Her bölge için sıcaklık hatası ve yük
durumundan (bölgedeki yük, sıcaklığın değişim yönü ve yaklaşan parti) oluşan 10×10'luk bir hücre seçilir
(README §4.2). Fiili ısıtıcı gücü ve bekleyen komutlar tabloya girmez, çünkü her yeni eksen tabloyu katlar.
Bu yüzden Q-tablosu açısından gecikme sorunu açık kalır; aşmanın ana sebebi budur (README §6).

**Ablasyon (E8):** `m_next`, `v_next` ve `t_arrival` çıkarıldığında Q-tablosunun test RMSE'si değişmiyor
(önizlemeli 24.08 ± 4.01, önizlemesiz 24.04 ± 1.01 °C; referans ayar). Tablo önizlemeyi kullanmayı öğrenemedi.

## 2. Eylem uzayı `A`

- Fiziksel büyüklük: `u ∈ [0, 1]^N`, her bölgenin rezistans güç oranı ya da doğalgaz vana açıklığı
- **Artımlı (velocity) eylem, varsayılan:** `a ∈ [−1, 1]^N`, `u_t = clip(u_{t−1} + 0.1 · a, 0, 1)`.
  Ajan güç seviyesini değil, güçteki değişimi seçer (adım başına en fazla ±%10).
- Mutlak eylem (`action_mode: absolute`): `u_t = (a + 1) / 2`. Yalnızca sabit güçlü testlerde kullanılır, bkz. §2.1.
- Fiziksel uygulama aktüatör gecikmesinden geçer (bkz. simülatör dokümanı)
- PID de aynı ±%10/adım sınırıyla çalışır (`power_to_action`), böylece kıyas adil kalır

### 2.1 Neden artımlı eylem?

Sebep fırının çok yüksek statik kazancı: `ΔT_ss = P_max · Δu / h ≈ 200 kW · 0.01 / 75 W/K ≈ 27 °C`.
Yani güç çıkışındaki %1'lik hata kalıcı durumda 27 °C sapma demek. Mutlak güç üreten bir ajanın
binde bir hassasiyetle çalışması gerekir. PID bu sorunu integral terimiyle doğal olarak çözer.
Artımlı eylemde kalıcı durumda ajanın yalnızca `a ≈ 0` üretmesi yeterli: integral davranış eylem
parametrizasyonuna gömülü. Bu, RL ile proses kontrolünde yaygın bir tasarım seçimidir.

## 3. Geçiş dinamiği `P`

Deterministik fizik ile stokastik bileşenlerin birleşimi:
- Deterministik: bölge enerji dengesi, bölgeler arası ısı iletimi, malzemeye ısı transferi,
  aktüatör/sensör gecikmeleri
- Stokastik: parti varış zamanları (Poisson), parti kütlesi ve hızı (düzgün dağılım), sensör
  gürültüsü (Gauss), ortam sıcaklığı (bölüm başında rastgele)

Denklemler ve parametreler: [03_simulator_tasarimi.md](03_simulator_tasarimi.md).

## 4. Ödül fonksiyonu `R`

Adım ödülü:

```
r_t = −( w₁ · (1/N) Σ eᵢ²  +  w₂ · E_t / E_max  +  w₃ · (1/N) Σ |uᵢ,t − uᵢ,t−1| )  −  c_safe · 1[max Tᵢ > T_max]

eᵢ    = (T_true_i − T_ref_i) / ΔT_scale         (ΔT_scale = 20 °C)
E_t   = Σ P_max · u_eff_i · Δt / 3.6e6           (bu adımdaki kWh)
E_max = N · P_max · Δt / 3.6e6                   (tüm ısıtıcılar tam güçteyken kWh)
```

Ağırlıklar: `w₁ = 1.0`, `w₂ = 0.2`, `w₃ = 0.05`, `c_safe = 2` (T_max üstünde geçen **her adım** için;
bölüm bitmez, bkz. §5.1).

**Neden bu biçim?**
- Ödev "sapma karesi ve enerji ile ters orantılı" istiyor. Negatif ağırlıklı maliyet bunu karşılar.
- Terimler normalize edildiği için ağırlıklar doğrudan "öncelik" anlamı taşır.
- Ödülde **gerçek** sıcaklık (`T_true`) kullanılır, gürültülü ölçüm değil. Gürültülü ölçüm
  kullanılsaydı ajan gürültüyü cezalandırılan bir şey olarak öğrenirdi. Simülasyonda bu
  ayrıcalığı kullanabiliyoruz; raporda belirtilmeli.

## 5. Reward shaping tartışması (rubrik maddesi)

Tasarım aşamasında beklenen yan etkiler:

| Durum | Beklenen yan etki |
|---|---|
| `w₂` çok büyük | Ajan ısıtıcıları kısar, sıcaklık hedefin altına iner (enerji tasarrufu ama kalite kaybı) |
| `w₂ = 0` | Enerjiyi umursamaz, PID'e benzer ya da daha agresif davranır |
| `w₃ = 0` | Eylemde titreşim (chattering), gerçekte vana/röle aşınması |
| Mutlak hata yerine kare hata | Büyük düşüşler çok cezalanır, küçük sürekli hatalar az cezalanır |
| Ölçülen sıcaklıkla ödül | Gürültü ödüle karışır, öğrenme yavaşlar |

Deney: `w₂ ∈ {0, 0.2, 1.0}` için RMSE–kWh grafiği ve `w₃ = 0` (bkz. deney dokümanı E4).

**Sonuç (E4, Q-tablosu referans ayarı, 3 seed):** İlk iki beklenti gerçekleşmedi. Enerji w₂'den neredeyse
bağımsız kaldı (673.7–686.5 kWh); tüketimin büyük kısmını duvar kayıpları ve parti ısıtması belirliyor. Beklenmeyen
sonuç RMSE'de: w₂ = 0'da 30.12, w₂ = 0.2'de 24.08, w₂ = 1'de 22.12 °C. Enerji cezası olmayan ajan ısıtıcıları
gereğinden az açtı. Enerji terimi kaba ızgarada eylemleri ayırt eden bir işaret gibi davranmış olabilir (test
edilmedi). Bu bulgu son ayarda w₂ = 1 seçilmesine yol açtı. Kare hatadan beklenen kalıcı ısı deposu
görülmedi: B1 varıştan önceki 10 dakikada hedefin 10.2–13.8 °C altında (PID −11.8 °C;
`results/eval/reward_analysis.md`). `w₃ = 0` ile titreşim artışı da gözlenmedi.

### 5.1 Ödül açığı: erken bitiş

`T > T_max` bölümü bitirseydi (`terminated`) ve tek seferlik `c_safe = 10` ceza verseydi, ödül hep negatif
olduğu için bölümü kısaltmak kârlı olurdu. `scripts/reward_analysis.py` bunu eğitimsiz ölçer: tam güç kuralı
bölümü ortalama 97. adımda bitirip −700 alıyor, PID tüm bölümde −886 alıyor.

Bu, maliyet biçimli (hep negatif) ödüllerde sonlandırmanın ödül olarak algılanmasının klasik
örneğidir. **Tasarım:** ihlal bölümü bitirmez, ihlal süren her adım `c_safe = 2` cezası alır.
Takip terimi de T_max'ta zaten `((950 − 850) / 20)² = 25` olduğundan kaçış yolu kalmaz. Değerlendirmede `unsafe_min` (T_max üstünde geçen dakika)
raporlanır.

## 6. Bölüm başlangıcı ve sonu

- **`reset`:** bölgeler hedef sıcaklık civarında (± 10 °C) dengede başlar, fırın boştur; parti
  takvimi seed'den üretilir.
- **`terminated`:** kullanılmıyor (her zaman `False`, bkz. §5.1).
- **`truncated`:** 720. adım (zaman sınırı; değer tabanlı yöntemler burada bootstrap eder).
