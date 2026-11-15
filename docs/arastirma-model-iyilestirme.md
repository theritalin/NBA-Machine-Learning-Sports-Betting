# Modeli iyileştirme araştırması (2026-11-15)

Üç araştırma ajanının (özellik/veri, modelleme/bahis yöntemi, depo içi analiz) raporlarının birleşimi.
Web ajanları çoğunlukla arama özetlerini okudu; "doğrulanamadı" işaretli maddeler birincil kaynaktan
teyit edilmedi. Depo içi bulgular koddan ve salt-okunur sorgulardan; hiçbiri çalıştırılarak sınanmadı.

## Başlangıç durumu

| Ölçüm (hiç görülmemiş 1.700 maç) | Model | Bahis şirketi (kapanış) |
|---|---|---|
| Maç sonucu isabeti | %65,8 | %68,2 |
| Log-loss | 0,620 | 0,594 |
| Alt/üst isabeti | %50,4 | başabaş için %52,4 gerekir |

Gerçekçi hedef: piyasayı geçmek değil, %67 civarı isabet ve iyi kalibre olasılık. Kaynaklar NBA
kapanış çizgisini genelde verimli buluyor (https://econpapers.repec.org/RePEc:buc:jgbeco:v:5:y:2011:i:2:p:42-56,
https://dc.etsu.edu/etsu-works/17780).

## Öncelik sırası (şefin sentezi)

Sıra: önce ucuz ve ölçümü düzelten işler, sonra yeni sinyal, en son yeni veri gerektirenler.

| # | İş | Beklenen etki | Zorluk | Değişen taraf |
|---|---|---|---|---|
| 1 | Ölçümü sağlamlaştır: yayındaki modeli walk-forward sına, log-loss/Brier, sezon bloklu bootstrap güven aralığı, alt/üst backtest'i | Tanı değeri yüksek (sonraki her deneyin hakemi) | Düşük-orta | backtest |
| 2 | Eğitim betiği düzeltmeleri: erken durdurma `mlogloss` ile, `binary:logistic`, sınıf ağırlığını kaldır, sığ ağaç aralığı (derinlik ≤5) | Orta-yüksek | Düşük | eğitim |
| 3 | Maç skorundan Elo (sayı farkı düzeltmeli) ve son 5/10 maç formu; veri depoda var (OddsData `Points`, `Win_Margin`) | Yüksek | Orta | veri seti + canlı + backtest |
| 4 | Özellik temsili: ev−deplasman farkı, gün içi z-skor; `_RANK`, `GP/W/L`, `MIN` çıkar; dört faktör ve verimlilik türet | Orta-yüksek | Düşük-orta | veri seti + canlı |
| 5 | Piyasa olasılığını (komisyonsuz) modele girdi ver; model yalnızca farkı öğrensin | Yüksek ama kanıtı zayıf | Orta | veri seti + canlı + backtest |
| 6 | Sayı farkı ve toplam sayı regresyonu, olasılığı dağılımdan üret (alt/üst için tek umut) | Orta | Orta | eğitim + canlı |
| 7 | Dinlenme/yorgunluk tutarlılığı: tek kırpma kuralı, ardışık maç, seyahat mesafesi | Düşük-orta | Orta | veri seti + canlı |
| 8 | Sakatlık/kadro özelliği (resmi rapor 2021-22'den beri) ve açılış oranı toplayıp CLV izleme | Potansiyel en yüksek | Yüksek, yeni veri gerekir | yeni veri hattı |

Not: 6. maddeden sonra seçilen ayarlarla tüm veride yeniden eğitim yapılmalı (yayındaki model
2023-03-27 sonrasını hiç görmedi); bunun için son sezon ayrı bekletme kümesi olarak ayrılmalı.

## Depo içi bulgular (koddan)

- **Erken durdurma yanlış metrikle:** `eval_metric: ["mlogloss","merror"]`; XGBoost sonuncuyu
  (merror) kullanır (`src/Train-Models/XGBoost_Model_ML.py:97`). Dokümantasyon bilgisi, çalıştırılarak doğrulanmadı.
- **Aynı küme hem durdurma hem ölçüm/kalibrasyon** (`XGBoost_Model_ML.py:128-130, 184-196`); 60-100
  denemenin en iyisini seçmek sonucu iyimser gösterir.
- **Sınıf ağırlığı** yalnız eğitimde uygulanıyor (`:70-77`); olasılıkları kaydırır. Alt/üstte seyrek
  "push" sınıfının ağırlığı çok büyüyor.
- **Seçilen model derin** (derinlik 9, 2194 tur); `notes.txt` ve backtest sığ ağaçların daha iyi olduğunu söylüyor.
- **Final model son %19 veriyi görmüyor** (2023-03-27 sonrası: tam iki sezon).
- **`backtest.py` yayındaki modeli değil**, sabit parametreli ayrı bir modeli sınıyor; güven aralığı yok.
- **Playoff:** takım tabloları playoff boyunca donuyor (yalnız normal sezon), ama oran tablolarında
  playoff maçları var; eğitime bayraksız giriyor.
- **Dinlenme günü tutarsız:** 2012-2023 satırlarında üst sınır 9, 2024 sonrasında 17'ye çıkıyor; sezon
  başı varsayılanı üç yerde farklı (7 / 10 / 7).
- **Veri boşlukları:** oran/skor verisi 2026-01-07'de bitiyor; 2023-24'te Şubat ayında yalnız 114 satır.
- **Takım eşlemesi sırayla** (`Create_Games.py:196-197`); denetlenen tablolar alfabetik, eskiler doğrulanamadı.
- **Kalıntılar:** `dataset.sqlite` içinde eski tablolar ve `odds_*_new` kopyaları; hangisinin geçerli olduğu belirsiz.
- Hesap (ajanın yorumu): 1.700 maçta %65,8 için %95 güven aralığı ±2,3 puan; alt/üst için ±2,4.

Mevcut veriden yeni indirme olmadan türetilebilenler: maç skoru ve Elo (ev puanı = (Points+Win_Margin)/2),
son N maç formu, ev/deplasman ayrı form, rakip puanı (PTS−PLUS_MINUS), possession ve verimlilik,
dört faktör, son 7 günde maç sayısı, dinlenme farkı, seyahat (30 şehir için koordinat tablosu gerekir).

## Modelleme ve bahis yöntemi (web)

- **Kalibrasyona göre model seçimi:** Walsh & Joshi (2024), NBA'de kalibrasyonla seçilen modelin
  isabetle seçilenden belirgin daha kârlı olduğunu bildiriyor (ortalama ROI +%34,7'ye karşı −%35,2);
  tek sezonluk deney, rakamlar sürümler arasında değişmiş.
  https://researchportal.bath.ac.uk/en/publications/machine-learning-for-sports-betting-should-model-selection-be-bas/
- **Sayı farkı regresyonu:** fark, spread etrafında yaklaşık normal (σ≈12; blog düzeyi sezgi).
  https://nfosignal.substack.com/p/basketball-the-random-walk — alt/üst için kaynaklı somut model: doğrulanamadı.
- **Piyasa verimliliği:** kapanış fiyatı tarafsız; toplam çizgilerinde sezon başı yanlılığı bildirilmiş
  (https://dc.etsu.edu/etsu-works/17780). Piyasa çizgisini girdi yapmanın kanıtlı kazancı: doğrulanamadı.
- **CLV:** hakemli kanıt bulunamadı; pozitif CLV kâr garantisi değil (https://www.karlwhelan.com/?p=2595).
  Açılış çizgimiz olmadığı için şu an ölçülemez.
- **Çoklu deneme düzeltmesi:** Deflated Sharpe Ratio (https://papers.ssrn.com/abstract=2460551),
  backtest aşırı uyum olasılığı (https://papers.ssrn.com/abstract=2326253).
- **Bahis boyutu:** parametre belirsizliğinde Kelly küçültülmeli (Baker & McHale 2013,
  https://econpapers.repec.org/RePEc:inm:ordeca:v:10:y:2013:i:3:p:189-199; Metel 2017, https://arxiv.org/pdf/1701.02814).
- Ajanın hesabı (yorum): +%1 getiriyi şanstan ayırmak için on binlerce bahis gerekir; 9 sezonluk
  −%14…+%26 dağılımı gürültüyle uyumlu.

## Özellikler ve veri kaynakları (web)

- **Elo:** K=20, sayı farkı çarpanı (MOV+3)^0.8 / (7.5+0.006·Elo farkı); ev avantajı 100 puan ve
  sezonlar arası 0.75·Elo+0.25·1505 taşıma yalnız üçüncü taraf koddan (birincil kaynak doğrulanamadı).
  https://luke-lite.github.io/blog/how-to-build-an-NBA-Elo-rating-system
- **Seyahat/ardışık maç:** eve dönüşte her ek 500 km kazanma olasılığını ~%4 düşürüyor
  (Charest ve ark. 2021, https://pmc.ncbi.nlm.nih.gov/articles/PMC8636381).
- **Sakatlık verisi:** `nbainjuries` paketi resmi raporu 2021-22'den itibaren çekiyor
  (https://pypi.org/project/nbainjuries/). Oyuncu etki metriklerinin (EPM, DARKO) ücretsiz indirilmesi: doğrulanamadı.
- **Alt/üst:** hakem ekibi etkisi için hakemli çalışma bulunamadı; tempo ve üçlük hacmi kanıtı doğrulanamadı.
- **Veri kaynakları:** `nba_api` (resmi olmayan, belgesiz uç noktalar, istekler arası 1-3 sn önerilir,
  https://github.com/swar/nba_api); Kaggle `wyattowalsh/basketball` (https://github.com/wyattowalsh/nbadb);
  the-odds-api ücretsiz ayda 500 kredi, geçmiş oranlar ücretli planda (https://the-odds-api.com/sports-odds-data/nba-odds.html).
  balldontlie, pbpstats, dunksandthrees, Basketball-Reference: araştırılmadı.
- **Tavan:** literatürde %57-73 arası dağınık isabet bildiriliyor (derleme: https://arxiv.org/pdf/2410.21484);
  dört faktör, kayan pencere ve rakım için kanıt: doğrulanamadı.

## Şefin düzeltmeleri ve bakiye

- İki web ajanı "sezon sonu ortalaması sızıntısı" riskini birinci sıraya koydu. Bu daha önce denetlendi:
  veri setindeki istatistikler maç öncesi (takımın galibiyet sayısı her maçta yalnız önceki maçın
  sonucuyla artıyor, tüm sezonlarda %100). 2024-25 sonrasındaki sızıntı düzeltilmişti.
- Ajanların web bulguları arama özetine dayanıyor; karar vermeden önce 3 ve 5. maddelerin kaynakları
  tam metinden okunmalı. arXiv 2606.27957 (ardışık maçta %44) ve Princeton tezi doğrulanmadı.
- Bu ortamdan NBA ve oran sitelerine erişim kapalı; 8. madde ve canlı doğrulama burada yapılamaz.
- Araştırma kod değiştirmedi; yukarıdaki hiçbir öneri henüz denenmedi.

Bu hukuki/finansal tavsiye değildir.
