# NBA Bahis Önerileri (Makine Öğrenmesi) 🏀

Bugünkü NBA maçları için takım istatistiklerinden kazanan ve alt/üst tahmini üretir, bunları bahis
sitesinin oranlarıyla karşılaştırır ve **hangi bahsin oynanmaya değer olduğunu** tek tabloda gösterir.

```
▶ ÖNERİLEN BAHİSLER (en avantajlıdan başlayarak)
  #  Maç                               Tür         BAHİS                    Oran         Kazanma  Beklenen kâr/100  Öneri      Yatır
  1  Denver Nuggets @ Detroit Pistons  Maç Sonucu  Detroit Pistons kazanır  7.50 (+650)    18.6%             +39.1  ★★★ GÜÇLÜ  15.04
```

## Kurulum ve kullanım

```bash
pip3 install -r requirements.txt
python3 main.py -odds=fanduel -kasa 1000
```

| Seçenek | Anlamı |
|---|---|
| `-odds=fanduel` | Oranlar otomatik çekilir (fanduel, draftkings, betmgm, pointsbet, caesars, wynn, bet_rivers_ny). Verilmezse oranları elle girersiniz; ondalık (1.85) ya da Amerikan (-118) yazabilirsiniz. |
| `-kasa 1000` | Toplam bahis bütçeniz. "Yatır" sütunu buna göre hesaplanır. |
| `-xgb` / `-nn` / `-A` | XGBoost (varsayılan) / sinir ağı / ikisi birden. Sinir ağı için TensorFlow gerekir (ayrıca kurulur). |
| `-season 2026-27` | Sezonu elle seçmek için (varsayılan: içinde bulunulan sezon). |
| `-json` | Sonuçları JSON olarak verir. |

## Çıktı nasıl okunur?

- **ÖNERİLEN BAHİSLER**: yalnızca oynanmaya değer bulunan bahisler, en avantajlısı en üstte. Liste boşsa en iyi karar o gün oynamamaktır.
- **Oran**: ondalık oran (parantez içinde Amerikan oranı). 2.10 = 100 yatır, kazanırsan 210 al.
- **Kazanma**: bahsin tutma ihtimali.
- **Beklenen kâr/100**: aynı bahsi çok kez oynasanız 100 birim başına ortalama kazanç. Eksi ise uzun vadede kaybettirir.
- **Öneri**: ★★★ GÜÇLÜ (beklenen kâr ≥ %15), ★★ ORTA (≥ %7). Altındakiler listelenmez.
- **Yatır**: çeyrek Kelly kriteriyle önerilen tutar; tek bahiste kasanın en fazla %3'ü.
- **TÜM MAÇLAR**: her maç için modelin favorisi, iki tarafın oranı ve beklenen kârı, alt/üst tahmini.

## Ne kadar güvenilir?

Kısa cevap: **model bahis şirketini yenemiyor; öneriler kanıtlanmış bir kazanç yöntemi değil.**

- Kullanılan maç sonucu modeli (maç skorlarından türetilen Elo ve son-form özellikleriyle), hiç görmediği
  1.697 maçta (Kasım 2024 – Ocak 2026) **%65,7** isabet etti, kalibre log-loss'u 0,6185 (özelliksiz önceki
  model 0,6207); aynı dönemde bahis şirketinin favorisi **%68,2** kazandı ve onun log-loss'u 0,5941.
  Yani model piyasanın hâlâ gerisinde.
- `python3 backtest.py` aynı stratejiyi 9 sezonda sınar (her sezon yalnızca önceki sezonlarla eğitilir,
  10.311 maç): ORTA+ bahislerin toplam getirisi −%3,7 (%95 güven aralığı −%11,5 ile +%5,9), GÜÇLÜ −%2,5;
  sezonlara göre ORTA+ −%23,4 ile +%17,8 arasında, 9 sezonun 5'i zararda. Yani sonuç şanstan ayırt
  edilemiyor. Olasılık kalitesi: modelin log-loss'u 0,6260, piyasanın 0,6020 (fark +0,0240, %95 güven
  aralığı +0,0199 ile +0,0275: model piyasadan anlamlı biçimde kötü). Elo/form özellikleri
  `--drop-features '^(Elo|Form|HomeSplit|AwaySplit|Games7|B2B)'` ile çıkarılınca aynı tarifle model
  log-loss'u 0,6325 (fark +0,0305; eski varsayılan tarifle 0,6305), yani özellikler olasılığı biraz iyileştiriyor
  ama piyasayı yakalamaya yetmiyor.
  Aynı komut log-loss/Brier'ı piyasayla karşılaştırır ve alt/üst backtest'ini de koşar
  (`--json YOL`, `--seasons N`, `--first-season YIL`, `--drop-features REGEX` seçenekleri var).
- Alt/üst modeli yazı-turadan farksız (bekletmede %49,2, backtest'te %50,0). Güveni neredeyse hep %50-52 olduğu için alt/üst bahsi
  pratikte hiç önerilmez.
- Bu yüzden modelin olasılığı tek başına kullanılmaz; bahis şirketinin (komisyonu çıkarılmış) olasılığıyla
  %30 model / %70 piyasa oranında harmanlanır.
- Model sakatlık ve kadro bilgisini bilmez.

Listeyi "modele göre en az kötü bahisler" olarak okuyun. Kaybetmeyi göze alamayacağınız parayla oynamayın.

## Web arayüzü

```bash
cd Flask
flask --debug run
```

Aynı tabloyu tarayıcıda üç bahis sitesi için gösterir; kasa tutarını sayfadan değiştirebilirsiniz.

## Yeni veri ve model eğitimi

Proje kökünden çalıştırın:

```bash
PYTHONPATH=. python3 src/Process-Data/Get_Data.py        # günlük takım istatistikleri
PYTHONPATH=. python3 src/Process-Data/Get_Odds_Data.py   # oranlar ve skorlar
PYTHONPATH=. python3 src/Process-Data/Create_Games.py    # eğitim veri setini oluşturur
python3 src/Train-Models/XGBoost_Model_ML.py --trials 60
python3 src/Train-Models/XGBoost_Model_UO.py --trials 40
```

Maç sonucu modeli ayrıca maç skorlarından türetilen 16 özellik kullanır (Elo, son 10/5 maç farkı, son 10 maç
galibiyet oranı, evde/deplasmanda sezon ortalaması, son 7 günde oynanan maç, dünkü maç): `Create_Games.py`
bunları veri setinin sonuna ekler, `main.py` aynı kodla (`src/Utils/Game_Features.py`) canlı hesaplar. Canlı
tahminde geçmiş `Data/OddsData.sqlite` + `Data/nba-<yıl>-UTC.csv` içindeki `Result` sütunundan okunur;
sonuçlar 3 günden eskiyse `main.py` uyarı basar. Bu özellikleri beklemeyen eski modeller (106 girdi) çalışmaya
devam eder. Karşılaştırma için: `python3 backtest.py --drop-features '^(Elo|Form|HomeSplit|AwaySplit|Games7|B2B)'`.

Eğitim hedefi isabet değil olasılık kalitesidir (log-loss). İki hedef de `binary:logistic` ile eğitilir
(alt/üstte push maçları atılır); arama zaman sıralı çapraz doğrulamayla yapılır, erken durdurma her katın
eğitim kısmının son %10'unda, skor ayrılmış doğrulama katında ölçülür. Kalibrasyon kat-dışı tahminlerden
öğrenilir; son %10 bekletme kümesinde ölçülür. Varsayılan olarak yayın modeli tüm veride yeniden eğitilir
(`--no-refit` ilk %90'lık modeli kaydeder). Çıktılar: `XGBoost_<bekletme isabeti>%_<ML|UO>_<YYYYMMDD>.json`
ile yanında `_calibration.json` ve `_meta.json` (bekletme metrikleri, parametreler, tur sayısı).

Sezon tarihleri `config.toml` içindedir. Tahmin sırasında `Models/XGBoost_Models` içindeki en yeni model
kendiliğinden seçilir. Dinlenme günü hesabı için `Data/nba-<yıl>-UTC.csv` fikstür dosyası kullanılır.

## Kaynak depoya göre farklar

Bu depo [kyleskom/NBA-Machine-Learning-Sports-Betting](https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting)
üzerine kuruludur (veri, eğitim betikleri, dinlenme günü özelliği oradan). Eklenenler ve düzeltilenler:

- Önerilen bahisler tablosu, Türkçe çıktı, kasa bazlı tutar önerisi, JSON çıktısı, sade web arayüzü.
- 2024-25 sonrasında eğitim verisine maçın kendi sonucu sızıyordu (duyurulan %68,9 isabet bundan
  kaynaklanıyordu); düzeltildi, veri seti yeniden kuruldu ve modeller temiz veriyle yeniden eğitildi.
- Canlı tahminde alt/üst modelinin girdileri eğitimdekinden farklı sıradaydı; düzeltildi.
- Olasılık kalibrasyonu tahmin sırasında sessizce devre dışı kalıyordu; artık JSON olarak saklanıp uygulanıyor.
- Sezon ve adresler elle güncellenmek zorunda değil; eksik oranlar programı çökertmiyor.
- Web arayüzündeki oyuncu/kadro pencereleri kaldırıldı (koda gömülü, başkasına ait bir API anahtarına bağlıydı).

## Testler

```bash
python3 -m unittest discover -s Tests -p 'test_*.py'
```
