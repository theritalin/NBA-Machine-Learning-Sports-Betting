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
| `-xgb` / `-nn` / `-A` | XGBoost (varsayılan) / sinir ağı / ikisi birden. Sinir ağı için TensorFlow gerekir. |
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

`python3 backtest.py` modeli geçmiş sezonlarda dürüst biçimde sınar (yalnızca önceki sezonlarla eğitip
sonraki sezonda bahis oynar, 2017-23, 6.679 maç):

- Model tek başına bahis şirketinden **daha az** isabetli (%64'e karşı %67). Bu yüzden öneriler modelin
  olasılığı ile bahis şirketinin (komisyonu çıkarılmış) olasılığının ortalamasına dayanır.
- Önerilen maç sonucu bahislerinin getirisi 6 sezonda toplam **+%1 ile +%4** arasında; sezonlara göre
  −%11 ile +%20 arasında oynuyor. Yani küçük ve garantisi olmayan bir avantaj.
- Alt/üst modeli yazı-turadan farksız; yalnızca güveni %70'in üzerindeyken az da olsa tutuyor
  (%70-80 güvende %54,5, %80 üstünde %60 isabet). Bu yüzden alt/üst bahsi yalnızca güven ≥ %80 iken önerilir.
- Model sakatlık, kadro, dinlenme günü gibi bilgileri bilmez ve 2022-23 sezonuna kadarki veriyle eğitilmiştir.

Kaybetmeyi göze alamayacağınız parayla oynamayın.

## Web arayüzü

```bash
cd Flask
flask --debug run
```

Aynı tabloyu tarayıcıda üç bahis sitesi için gösterir; kasa tutarını sayfadan değiştirebilirsiniz.

## Yeni veri ve model eğitimi

```bash
cd src/Process-Data
python -m Get_Data        # sezon/yıl dosyanın başında elle ayarlanır
python -m Create_Games
cd ../Train-Models
python -m XGBoost_Model_ML
python -m XGBoost_Model_UO
```

## Testler

```bash
python3 -m unittest Tests/Expected_Value_Test.py Tests/Recommender_Test.py
```
