# 01 – Gereksinim Analizi

## 1. Problem tanımı

Sürekli tavlama fırını birden çok ısıtma bölgesinden oluşur (ör. ön ısıtma, ısıtma, tutma). Her
bölgede bir termokupl sıcaklığı ölçer, bir rezistans ya da doğalgaz brülörü de ısı verir. Fırına
soğuk bir parti girdiğinde malzeme bulunduğu bölgeden hızla ısı çeker ve bölge sıcaklığı düşer
(termal çöküş).

Klasik PID:
- yalnızca hata oluştuktan sonra tepki verir,
- yüksek ısıl eylemsizlik ve ölü zaman yüzünden geç kalır,
- geç kalınca agresif davranır, bu da aşma (overshoot) ve fazla enerji tüketimi demektir.

Fırına girmek üzere olan partinin kütlesi ve hızı tesiste önceden bilinir (izleme sistemi). RL ajanı
bu bilgiyi kullanarak ön ısıtma yapabilir. Ödevin RL'e bıraktığı fırsat budur.

## 2. Kapsam

**Kapsam içi**
- Fiziği yaklaşık temsil eden, `gymnasium.Env` uyumlu çok bölgeli fırın simülatörü
- Bölge başına PID baseline
- Sinir ağı içermeyen model tabanlı RL ajanı: doğrusal öğrenilmiş model + CEM planlama (ana yöntem, MB-RL)
- Karşılaştırma için sinir ağı içermeyen modelden bağımsız yöntemler: tile coding'li Q-öğrenme, LSPI, ARS
- Çok seed'li eğitim, hiperparametre analizi, gürültü altında gürbüzlük testi
- Rapor

**Kapsam dışı**
- 3B görselleştirme, CFD, gerçek fırın verisi ya da donanım
- Malzemenin metalurjik iç yapı modeli (yalnızca kütle × özgül ısı)
- Yanma kimyası (doğalgaz yalnızca "orantılı güç" olarak modellenir)

**Kısıt**
- Derin pekiştirmeli öğrenme kullanılmaz: sinir ağı, PyTorch vb. yok. Ödev metninin önerdiği TD3 ve PETS sinir ağlı
  olduğu için önerinin "Model-Based RL / MPC-RL" kolu doğrusal modelle uygulanır.
- PID yalnızca baseline'dır; hibrit (PID + RL) kullanılmaz.

## 3. Fonksiyonel gereksinimler

| ID | Gereksinim |
|---|---|
| FR-1 | Ortam N bölgeyi (varsayılan 3) birbiriyle etkileşen ısıl kütleler olarak simüle etmeli. |
| FR-2 | Soğuk partiler rastgele zaman, kütle ve hızla gelmeli, bölgeler arasında hareket edip ısı çekmeli. |
| FR-3 | Gelecek partinin kütlesi, hızı ve varışa kalan süre gözlemde olmalı (önizleme bilgisi). |
| FR-4 | Isıtıcıda gecikme (aktüatör zaman sabiti + ölü zaman), termokuplda ölçüm gecikmesi ve gürültü olmalı. |
| FR-5 | Eylem: her bölge için `[0, 1]` aralığında sürekli güç oranı. |
| FR-6 | Ödül: sıcaklık sapmasının karesi ve harcanan enerji (kWh) ile ters orantılı olmalı, ağırlıklar ayarlanabilmeli. |
| FR-7 | Her adımda enerji (kWh) ve bölge sıcaklıkları `info` içinde raporlanmalı. |
| FR-8 | Aynı seed ile aynı senaryo tekrar üretilebilmeli (adil PID–RL kıyası için sabit test seti). |
| FR-9 | PID baseline aynı ortam arayüzü üzerinden çalışmalı. |
| FR-10 | Değerlendirme betiği PID ve RL ajanlarını aynı test senaryolarında koşturup metrik tablosu ve grafik üretmeli. |

## 4. Fonksiyonel olmayan gereksinimler

| ID | Gereksinim |
|---|---|
| NFR-1 | **Tekrarlanabilirlik:** tüm deneyler seed'li; ayarlar `configs/` altında YAML dosyalarında. |
| NFR-2 | **Donanım:** standart bir dizüstünde, CPU ile çalışmalı. Ana yöntemin bir seed'lik eğitimi birkaç saati geçmemeli. |
| NFR-3 | **Fiziksel güvenlik:** sıcaklık üst sınırı (ör. 950 °C) aşılırsa ağır ceza ve bölüm sonu. |
| NFR-4 | **Gürbüzlük:** ajan sensör gürültüsü ve eğitim aralığı dışındaki kütle/hız değerleri altında test edilmeli. |
| NFR-5 | **Okunabilirlik:** ortam kodu kısa ve açık olmalı; fizik denklemleri dokümandakiyle birebir eşleşmeli. |
| NFR-6 | **Test:** ortam için temel birim testleri (enerji korunumu yönü, sınırlar, seed tekrarı). |

## 5. Notlandırma eşlemesi (hocanın rubriği)

| Rubrik maddesi | Ağırlık | Karşılayan iş | Sprint |
|---|---|---|---|
| Matematiksel formülasyon (S, A, P, R + reward shaping tartışması) | %20 | `02_mdp_tasarimi.md` → rapor bölümü, ödül ablasyonu | 1, 3 |
| Simülasyon ortamı (`gymnasium.Env`, `reset`/`step`) | %25 | `furnace_env.py`, `scenarios.py`, testler | 1 |
| Klasik yöntem ile karşılaştırma (kıyas tablosu) | %25 | `pid.py`, `evaluate.py`, PID–RL tablosu ve grafikleri | 1, 2 |
| Eğitim, hiperparametre analizi, kararlılık (≥3 seed, gürültü testi) | %30 | `agents/mbrl.py`, seed'li eğitim, gürültü testi | 2, 3 |

## 6. Riskler

| Risk | Etki | Önlem |
|---|---|---|
| Simülatör çok kolay olur, PID zaten mükemmel çalışır | RL'in katkısı görünmez | Ölü zaman ve gecikmeyi gerçekçi tut; hedef: PID ile 1 tonluk partide 30–80 °C düşüş |
| Simülatör çok zor ya da kararsız olur | Ajan öğrenemez | Sprint 1'de rastgele ve PID politikalarıyla kalibre et |
| Ödül ağırlıkları kötü seçilir | Ajan ısıtıcıyı kapatır ya da enerjiyi umursamaz | Normalize edilmiş ödül terimleri, Sprint 3'te `w₂` taraması |
| Eğitim kararsız | Seed'ler arası büyük varyans | Gözlem normalizasyonu, ödül ölçekleme, ≥3 seed |
| Zaman yetmez | Eksik teslim | Çekirdek (env + PID + MB-RL + 3 seed + gürültü) önce |
| PID haksız biçimde zayıf ayarlanır | Kıyas güvenilmez olur | PID kazançlarını aynı simülatörde ızgara arama ile ayarla, yöntemi raporla |
