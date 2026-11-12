# Çok Bölgeli Tavlama Fırınında Q-Learning ile Sıcaklık Kontrolü

Bu proje, pekiştirmeli öğrenme dersi kapsamında yaptığım bir tablolu Q-Learning uygulamasıdır. Üç bölgeli bir tavlama
fırınına bant üzerinde soğuk metal partileri giriyor ve her parti girdiğinde bulunduğu bölgenin sıcaklığı düşüyor.
Fırını yöneten ajan, ısıtıcıların gücünü sabit bir kuralla değil, deneme yanılmayla ve aldığı cezalardan öğrenerek
ayarlıyor. Eğitilen ajanı, sanayide yaygın kullanılan klasik **PID** kontrolcüyle karşılaştırdım. Dersin şartı gereği
derin pekiştirmeli öğrenme kullanmadım; projede sinir ağı yok.

**Kısaca, konuyu hiç bilmeyenler için:** Bir fabrikada metal parçaları uzun bir fırının içinden geçiyor ve fırının her
bölümünün belirli bir sıcaklıkta kalması gerekiyor. İçeri soğuk metal girince fırın soğuyor. Bilgisayarda bu fırının
bir benzetimini yaptım ve ısıtıcıları ne kadar açacağına karar veren bir program (ajan) yazdım. Programa doğru cevap
öğretilmiyor: binlerce kez deniyor, sıcaklık hedeften uzaklaştığında ceza alıyor ve hangi durumda ne yapacağını kendisi
öğreniyor. Sonuç: program fırını yönetmeyi öğrendi, ama klasik yöntemi geçemedi; nedenini aşağıda gösteriyorum.

**Video anlatım:** GELCEK https://drive.google.com/drive/folders/1vY-wTDswQYrG6xvKbn3MynFxJGUSuJ-4

![Fırın animasyonu](results/figures/firin_animasyon.gif)

Aynı senaryoda üstte PID, altta benim eğittiğim Q-tablosu fırını yönetiyor. Koyu bloklar metal partileri; bir bölüm
maviye döndüğünde soğumuş, kırmızıya döndüğünde fazla ısınmış demek. Turuncu çubuk ısıtıcının ne kadar açık olduğunu
gösteriyor.

## Çalıştırma

```bash
pip install -r requirements.txt && pip install -e .
python -m pytest -q                                     # 18 test
python scripts/compare_e3.py --runs results/models/q_son_seed0 results/models/q_son_seed1 results/models/q_son_seed2 --tag e3
```

Son komut, depodaki eğitilmiş tablolarla PID karşılaştırmasını yaklaşık 1 dakikada yeniden üretir. Tek bir eğitim
yaklaşık 45 dakika sürer (`python -m furnace_rl.agents.q_table --seed 0 --config configs/q_table_son.yaml --env-config configs/env_w_energy_1.0.yaml`);
tüm eğitim ve grafikler için `scripts/run_experiments.sh` ve `scripts/run_evaluations.sh`.

## Problem Tanımı

**Tavlama**, metali belirli bir sıcaklığa ısıtıp bir süre orada tutarak dayanıklı ve kolay işlenebilir hale getirme
işlemidir. Fırın üç bölgeden oluşuyor, partiler soldan girip sağa ilerliyor:

```
              B1 ön ısıtma        B2 ısıtma           B3 tutma
           +-----------------+-----------------+-----------------+
 parti --> |  hedef 750 °C   |  hedef 850 °C   |  hedef 850 °C   | --> çıkış
           |  en çok 250 kW  |  en çok 200 kW  |  en çok 160 kW  |
           +-----------------+-----------------+-----------------+
```

Partiler 300–1500 kg arasında, rastgele zamanlarda geliyor. Bir parti girince bulunduğu bölgeden ısı çekiyor ve
sıcaklık düşüyor (termal çöküş).

Ajanın hedefi:

- Her bölgenin sıcaklığını hedefte tutmak
- Enerji tüketimini artırmamak
- 950 °C güvenlik sınırını hiç aşmamak

Zorluk fırının yavaşlığı: ısıtıcıya komut verildiğinde 30 saniye hiçbir şey olmuyor (ölü zaman), ısıtıcı yeni gücüne
birkaç dakikada ulaşıyor, sensör de geç ve gürültülü ölçüyor. Duşta musluğu çevirince suyun hemen ısınmaması gibi; fırın
bunun çok daha yavaşı. Öte yandan gelecek partinin bilgisi 10 dakika önceden biliniyor (önizleme) ve ajan bunu görüyor.

Gerçek fırında binlerce deneme yapılamayacağı için fırını bilgisayarda `gymnasium.Env` olarak simüle ettim
(`src/furnace_rl/envs/furnace_env.py`). Parametreleri gerçek bir fırından almadım; problem ne çok kolay ne imkânsız
olsun diye kalibre ettim.

## Aksiyonlar

Ajanın tek işi her bölgenin ısıtıcısını ne kadar açacağına karar vermek. 6 seçeneği var:

| Aksiyon | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Isıtıcı gücü | 0 (kapalı) | 0,2 | 0,3 | 0,4 | 0,6 | 1 (tam) |

Güç seçilen seviyeye bir anda değil, adım başına en çok %10 değişerek gidiyor; gerçek bir ısıtıcı da bir anda açılıp
kapanmaz. PID de aynı sınırla çalışıyor. Ajan 30 saniyede bir karar veriyor, çünkü bir komutun etkisi ancak o kadar
sonra görülmeye başlıyor.

## State Tasarımı

State (durum), ajanın karar verirken baktığı bilgidir; araba kullanırken göstergelere bakmanız gibi. Bu projede en çok
uğraştığım konu oldu, çünkü tablolu Q-Learning'de tablonun her satırı bir durum ve ajanın her satırı yeterince sık
görüp öğrenebilmesi gerekiyor.

Simülatör her adımda 30 sayı veriyor (sıcaklıklar, değişim hızları, yükler, ısıtıcı güçleri, gelecek parti...). Hepsini
tabloya koysaydım, her birini 10 kutuya bölsem bile 10³⁰ durum olurdu; bu ne bellekte tutulabilir ne öğrenilebilir.
Bu yüzden her bölgeye küçük bir tablo verdim ve state'i iki bilgiye indirdim:

```
state = (sıcaklık_hatası_kutusu, yük_durumu)      →  10 × 10 = 100 durum
```

| Sıcaklık hatası kutusu (ölçülen − hedef) | Yük durumu |
|---|---|
| < −100 · −100…−60 · −60…−30 · −30…−15 · −15…−8 | Yük yok: sıcaklık düşüyor / sabit / yükseliyor |
| −8…−3 · **−3…3 (hedefte)** | Yük yok: küçük / büyük parti yaklaşıyor |
| 3…8 · 8…20 · > 20 °C | Hafif yük / ağır yük (sıcaklık yükseliyor mu?), yük var ve büyük parti yaklaşıyor |

Kutular hedefin yakınında dar, çünkü ince ayar orada gerekiyor. Tablo bölge başına 100 durum × 6 aksiyon = 600 sayı,
üç bölgede toplam **1800 sayı**.

### Bu tasarıma nasıl vardım

İlk denemelerde daha büyük tablolar kullandım ama ajan öğrenemedi. Tablo küçüldükçe her hücre daha sık güncellendi:

| Durum temsili | Hücre (bölge başına) | Doğrulama sapması |
|---|---|---|
| 11 hata × 5 hız × 4 yük × 4 yaklaşan yük | 9 680 | 73,5 °C |
| 14 × 5 × 3 × 3 | 6 930 | 24,8 °C |
| **10 hata × 10 yük durumu** | **600** | **13,3 °C** |
| 10 hata × 10 değişim hızı (yük yok) | 600 | 22,3 °C |
| PID (karşılaştırma) | – | 11,9 °C |

Yük bilgisi önemli: bölge, içeride soğuk parti olduğu için de (tam güç gerekir), yüksüzken ısıtıcı kısık kaldığı için de
(biraz güç yeter) soğuyabilir. Yükü görmeyen tablo ikisini ayırt edemiyor. Seçimleri yalnızca doğrulama senaryolarıyla
yaptım, test senaryolarını en sona sakladım.

### Bu tasarımın bedeli

State'te ısıtıcının o anki gerçek gücü yok; eklemek tabloyu katlıyordu. Bu yüzden ajan ısıtıcının zaten tam güçte
olduğunu bilmiyor ve parti çıkınca sıcaklık hedefin üstüne çıkabiliyor.

## Reward Fonksiyonu

Ödül, ajana yaptığının iyi mi kötü mü olduğunu söyleyen puandır. Fırında "kazandım" denecek bir an olmadığı için ödül
her 10 saniyede veriliyor ve hep sıfır ya da negatif; yani bir ceza puanı. Formül kısaca şunu söylüyor: sıcaklık
hedeften uzaklaştıkça, enerji harcandıkça ve ısıtıcı sık açılıp kapandıkça ceza artıyor.

```math
r_t = -\left( w_1 \frac{1}{N}\sum_{i} e_{i,t}^2 + w_2 \frac{E_t}{E_{\max}} + w_3 \frac{1}{N}\sum_{i} |u_{i,t} - u_{i,t-1}| \right) - c\,\mathbf{1}\left[\max_i T_{i,t} > 950\,^\circ\mathrm{C}\right]
```

| Terim | Ağırlık | Anlamı |
|---|---|---|
| Sıcaklık sapması, e = (T − hedef) / 20 °C, karesi | w₁ = 1 | 10 °C sapma 0,25, 40 °C sapma 4 ceza: büyük düşüşler sert cezalanıyor |
| Enerji | w₂ = 0,2 | Harcanan enerji / tam güçteki enerji |
| Güç değişimi | w₃ = 0,05 | Isıtıcının sürekli açılıp kapanmasını önlemek |
| 950 °C üstü | c = 2 | Aşımın sürdüğü her adım için |

### Reward Tasarımında Aldığım Dersler

- **Bölümün ne zaman bittiği de ödülün parçası.** Sınır aşılınca bölüm bitseydi, ödül hep negatif olduğu için erken
  bitirmek kârlı olurdu: tam güç veren basit bir kural bölümü erken bitirip −700 alırken düzgün çalışan PID −886
  alıyor. Bu yüzden sınır aşımı bölümü bitirmiyor, her adım cezalanıyor.
- **Enerji ağırlığı beklediğim gibi davranmadı.** w₂'yi büyütünce enerjinin azalmasını bekliyordum; enerji neredeyse
  değişmedi (674–687 kWh), ama sıcaklık takibi iyileşti (w₂ = 0'da 30,1, w₂ = 1'de 22,1 °C). Son ayarda w₂ = 1 kullandım.
- **Güç değişimi cezasını kaldırınca titreşim artmadı;** ajan bir seviyeyi 30 saniye tuttuğu için zaten düşüktü.

## Q-Learning Algoritması

Q-tablosunu bir kopya kağıdı gibi düşünebilirsiniz: her durumda her seçeneğin ne kadar iyi olduğunu yazan bir tablo.
Ajan bir seçeneği deneyip cezasını görünce o hücredeki sayıyı biraz düzeltiyor; binlerce denemeden sonra tablo her
durumda en iyi seçeneği gösteriyor. Düzeltme klasik Q-Learning formülüyle yapılıyor; ajan 30 saniyede bir karar
verdiği için 3 adımın ödülü toplanıyor (k = 3):

```math
Q(s,a) \leftarrow Q(s,a) + \alpha(s,a)\left[R + \gamma^{k} \max_{a'} Q(s',a') - Q(s,a)\right]
```

Ajan hep en iyi bildiğini seçseydi hiç denemediği seçenekleri öğrenemezdi; bu yüzden ε olasılıkla rastgele bir seçenek
deniyor (epsilon-greedy). Kod: `src/furnace_rl/agents/q_table.py`.

### Hiperparametreler

| Parametre | Değer | Notum |
|---|---|---|
| Öğrenme oranı (α) | 0,5 → 0,02 | Hücre ne kadar çok güncellendiyse o kadar küçük adımla düzeltiliyor |
| İndirim faktörü (γ) | 0,99 | Yaklaşık 17 dakika ilerisini hesaba katıyor |
| Keşif oranı (ε) | 0,3 → 0,02 | Daha fazla keşif (0,5) sonucu kötüleştirdi |
| Başlangıç Q değeri | −5 | Ödüller negatif; 0 olsaydı denenmemiş seçenekler en iyi görünürdü |
| Bölüm sayısı | 5000 | 3,6 milyon adım; 3 farklı seed ile tekrarlandı |

## Eğitim Süreci

Her bölüm rastgele bir parti takvimiyle 2 saatlik bir fırın çalışması. Ajan 5000 bölüm boyunca dener ve tablosunu
günceller; her 50 bölümde bir 5 doğrulama senaryosunda denenir ve en iyi tablo saklanır. Aynı eğitim 3 farklı seed
(rastgeleliğin başlangıç değeri) ile tekrarlandı ki sonucun tesadüf olmadığı görülsün.

**PID**, fabrikalarda en çok kullanılan otomatik kontrol yöntemidir. Hiçbir şey öğrenmez; sıcaklık hedeften ne kadar
uzaksa o kadar güç veren sabit bir kuraldır. Ayarlarını 90 kombinasyonu deneyerek seçtim (`src/furnace_rl/baselines/pid.py`).

## Sonuçlar

### Öğrenme eğrisi

![Öğrenme eğrisi](results/figures/e2_rmse.png)

Başta tablo rastgele olduğu için sapma 172 °C. Ajan ilk birkaç yüz bölümde hızla iyileşiyor (500. bölümde 32 °C) ve
eğitimin sonunda 15 °C civarına iniyor. Üç seed birbirine çok yakın, yani öğrenme kararlı; ama kesikli çizgideki PID'in
(11,9 °C) altına inmiyor.

### Q-Learning ve PID karşılaştırması

Eğitimde hiç kullanılmayan 20 test senaryosunda. **Ortalama sapma (RMSE)**, sıcaklığın hedeften ortalama kaç derece
uzak kaldığı; küçük olan iyi.

| Metrik | PID | Q-tablosu (3 seed) |
|---|---|---|
| Ortalama sapma (RMSE) | **19,2 °C** | 22,4 ± 1,0 °C |
| En büyük düşüş | **68,4 °C** | 70,9 °C |
| Aşma (hedefin üstüne çıkma) | **16,1 °C** | 27,6 °C |
| Enerji | 690 kWh | **686 kWh** |
| Isıtıcı titreşimi | 0,052 | **0,028** |
| 950 °C ihlali | 0 | 0 |

Q-tablosu PID'i geçemedi. Buna karşılık aynı enerjiyi harcıyor, ısıtıcıyı yarı yarıya daha az oynatıyor ve güvenlik
sınırını hiç aşmıyor.

### Öğrenilen politika

![Öğrenilen politika](results/figures/politika_b1.png)

Tablolu Q-Learning'in güzel yanı, öğrenilenin doğrudan okunabilmesi. Şekil, 1. bölgenin tablosunda her durumda seçilen
gücü gösteriyor. Ağır yük altında ve hedefin çok altındayken tam güç veriyor; mantıklı. Ama yüksüz ve hedefteyken gücü
**0**'a çekiyor, biraz soğuyunca **0,4**'e çıkıyor. Sıcaklığı sabit tutan güç yaklaşık 0,35, yani iki seviyenin
arasında; ajan ikisi arasında gidip geliyor ve sıcaklık salınıyor. PID ise 0,35'i tam verebiliyor.

Özetle PID'i geçememesinin üç sebebi var:

1. **Kaba ızgara:** Doğru güç iki seviyenin arasında kalıyor.
2. **Isıtıcının gerçek gücü tabloda yok:** Ajan ısıtıcının zaten tam güçte olduğunu bilmiyor; parti çıkınca sıcaklık
   hedefin üstüne çıkıyor.
3. **Önizleme kullanılamadı:** "Parti yaklaşıyor" durumu tabloda çok seyrek görülüyor; ajan partiyi önceden görüp
   fırını hazırlamayı öğrenemedi.

### Diğer deneyler

| Deney | Sonuç |
|---|---|
| Sensör gürültüsü 0 → 5 °C | Q-tablosu PID'den daha çok etkileniyor (5 °C'de 26,0'a karşı 20,3 °C) |
| Tek ağır parti (1500 kg) | Q-tablosu PID'den iyi (19,2'ye karşı 21,0 °C): hemen tam güce geçiyor |
| Yavaş bant | Q-tablosu kötü: hedefte uzun süre durmak gerekince salınım baskın |
| Gelecek parti bilgisi olmadan eğitim | Sonuç değişmedi (24,0'a karşı 24,1 °C) |

## Karşılaştığım Zorluklar

- **Tabloyu ne kadar küçültmeli:** Büyük tablo öğrenilemedi, küçük tablo ince kontrol yapamıyor. 10 × 10 ortayı buldu
  ama PID'i geçmeye yetmedi.
- **Simülatörü ayarlamak:** İlk ayarlarda problem çok kolaydı, gücü azaltınca bu kez fırın yetersiz kaldı. Çözülebilir
  ama zor bir ortayı bulmak birkaç deneme sürdü.
- **Sonucun PID'in gerisinde kalması:** Bir iyileştirme turu yaptım (farklı güç seviyeleri, 30 s karar, w₂ = 1); sapma
  24,1'den 22,4 °C'ye indi ama PID'e ulaşmadı. Sonucu olduğu gibi yazdım.

## Terimler (basitçe)

| Terim | Anlamı |
|---|---|
| Ajan | Karar veren program; burada ısıtıcıları yöneten |
| State / aksiyon / ödül | Ajanın baktığı bilgi / yaptığı seçim / aldığı puan |
| Q-tablosu | Her durumda her seçeneğin ne kadar iyi olduğunu tutan tablo |
| Bölüm (episode) | Baştan sona bir deneme; burada 2 saatlik fırın çalışması |
| PID | Fabrikalarda yaygın, öğrenmeyen klasik kontrol yöntemi |
| Ölü zaman | Komuttan sonra hiçbir etkinin görülmediği süre (30 s) |
| Önizleme | Gelecek partinin 10 dakika önceden bilinmesi |
| RMSE | Sıcaklığın hedeften ortalama uzaklığı; küçük olan iyi |
| Seed | Rastgeleliğin başlangıç değeri |

## Proje Yapısı

```
├── configs/                  ayarlar (Q-tablosu, PID, test senaryoları)
├── src/furnace_rl/
│   ├── envs/furnace_env.py   fırın simülatörü (gymnasium.Env)
│   ├── agents/q_table.py     Q-tablosu, state, epsilon-greedy, eğitim
│   └── baselines/pid.py      PID
├── scripts/                  deney, grafik ve animasyon betikleri
├── tests/                    18 test (biri kodda sinir ağı olmadığını denetler)
├── docs/                     tasarım notları
└── results/                  eğitilmiş tablolar, sonuç tabloları, şekiller
```

## Sınırlamalar

- Simülatör parametreleri gerçek bir fırından değil, kalibrasyondan geliyor.
- 6 güç seviyesi ve 10 hata kutusu, PID'in sürekli gücü kadar ince kontrol yapamıyor.
- Her bölge kendi tablosunu bağımsız öğreniyor; 3,6 milyon adımlık eğitim gerçek fırında yaklaşık 417 gün eder.

## Sonuç

Bu projede derste gördüğümüz tablolu Q-Learning'i gerçekçi bir endüstriyel problemin benzetimine uyguladım. Ajan fırını
kontrol etmeyi öğrendi: aynı enerjiyle, daha az titreşimle ve güvenlik sınırını hiç aşmadan çalışıyor. Ama ortalama
sapmada PID'i geçemedi (22,4'e karşı 19,2 °C).

En çok şunu öğrendim: tablolu Q-Learning'de algoritma kadar state tasarımı önemli. Tablo büyürse öğrenilemiyor,
küçülürse ince kontrol yapılamıyor; öğrenilen tabloya bakarak bunun nedenini doğrudan görebildim.
