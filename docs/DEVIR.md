# Devir dosyası

Son güncelleme: 2026-11-15

## Açık işler

1. **Model iyileştirme deneyleri** — raporun (`docs/arastirma-model-iyilestirme.md`) 1-3. maddeleri yapıldı,
   denetlendi (aşağıda "Kapananlar"). Sıradakiler karar bekliyor: 4 (özellik temsili), 5 (piyasa olasılığını
   girdi yapmak), 6 (sayı farkı/toplam sayı regresyonu), 7 (dinlenme tutarlılığı), 8 (sakatlık verisi).
2. **Canlı doğrulama yapılmadı** — bu ortamdan stats.nba.com ve sportsbookreview.com erişimi kapalı;
   `python3 main.py -odds=fanduel` gerçek veriyle hiç koşmadı. Sinir ağı modeli de (TensorFlow yok) denenmedi.
3. **Veri 2026-01-07'de bitiyor** — `Get_Data` / `Get_Odds_Data` ile güncellenmeli (ağ erişimi gerekir).
4. **Depodaki özel ajanlar (`.claude/agents`) bu oturumda yüklenmedi**; araştırma genel amaçlı ajanla
   (sonnet) yapıldı. Yeni oturumda yüklenmeleri beklenir.
5. **Koda gömülü RapidAPI anahtarı** kaynak deponun geçmişinde duruyor (dosya bizde silindi); bize ait değil.
6. **Sonuç boşluğu 2026-01-08 … 2026-10-19** — depoda bu aralıktaki maç sonuçları yok (OddsData 2026-01-07'de
   bitiyor, `nba-2025-UTC.csv`'de yalnız 33 sonuç var). Elo/form geçmişi bu boşlukta donuyor: 2025-26 sezonunun
   son ~3 ayı Elo'ya hiç işlenmedi, 2026-27 başında yalnızca sezon geçişi (%25 ortalamaya çekme) uygulanıyor.
7. **Canlı Elo/form güncel sonuç ister** — `main.py` geçmişi OddsData + `Data/nba-<yıl>-UTC.csv` `Result`
   sütunundan okur; sonuçlar 3 günden eskiyse stderr'e uyarı basar. CSV'nin `Result` sütunu (ve OddsData)
   düzenli güncellenmeli; yoksa Elo/form bayatlar ve tahmin zayıflar.

8. **Denetçi cilaları (yapılmadı)** — (a) Elo modülündeki iki kural README'de anlatılmıyor: 90 günden uzun
   arada ortalamaya çekme, 0 farklı skorun beraberlik sayılması (OddsData'da 2012-12-26 Phoenix–Knicks 95-95
   muhtemelen veri hatası); (b) README'deki 0,6305 ve 0,5941 rakamları depodan yeniden üretilemiyor;
   (c) bayatlık uyarısı sezon içi uzun boşluğu yakalamıyor; (d) README yayındaki modelin tüm veride
   yeniden eğitildiğini (bekletme sayıları yeniden eğitimden önceki) açıkça söylemiyor.
9. **Dinlenme günü / ardışık maç hesabı dört ayrı yerde** (Get_Odds_Data, Add_Days_Rest, main.rest_days,
   Game_Features) ve kuralları farklı; birleştirilmedi.
10. **2023-24 için 60 maç eksik** — `odds_2023-24_new` (1195 maç) seçiliyor, `2023-24` tablosunda 1255 maç var.
11. **Alt/üst modeli sinyal taşımıyor** (%49-50); fiilen bahis önerilmiyor. Kapatılacak mı, regresyonla
    yeniden mi denenecek: karar bekliyor.
12. **Logistic_Regression_* ve NN_Model_* betikleri** eski eğitim hatalarını (sınıf ağırlığı, ortak
    kalibrasyon/test kümesi, 3 sınıflı alt/üst) hâlâ taşıyor; kullanılmıyorlar.

## Kapananlar

- Kaynak depo (kyleskom) birleştirildi, veri sızıntısı düzeltildi, modeller temiz veriyle yeniden eğitildi.
- `master` üzerindeki orkestra kurulumu çalışma dalına alındı.
- Ölçüm güçlendirildi: `backtest.py` log-loss/Brier, sezon bloklu bootstrap güven aralığı, alt/üst testi (denetim ONAY).
- Eğitim betikleri düzeltildi: ikili hedef, log-loss ile erken durdurma, sızıntısız CV, kat-dışı kalibrasyon,
  meta dosyası (denetim ONAY). Tek başına ölçülebilir kazanç getirmedi (bekletme 0,6213'e karşı 0,6207).
- Elo ve son form özellikleri eklendi, maç sonucu modeli yeniden eğitildi (denetim ONAY, sızıntı bağımsız
  hesapla sınandı). Backtest model log-loss 0,6325 → 0,6260 (piyasa 0,6020); bekletme 0,6207 → 0,6185.
  Model hâlâ piyasanın gerisinde; önerilen bahis getirisi −%3,7, güven aralığı [−%11,5, +%5,9].
