# Devir dosyası

Son güncelleme: 2026-11-15

## Açık işler

1. **Model iyileştirme deneyleri** — araştırma bitti (`docs/arastirma-model-iyilestirme.md`), uygulama
   başlamadı. Karar bekliyor: hangi maddelerle başlanacak (öneri: 1 → 2 → 3).
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

## Kapananlar

- Kaynak depo (kyleskom) birleştirildi, veri sızıntısı düzeltildi, modeller temiz veriyle yeniden eğitildi.
- `master` üzerindeki orkestra kurulumu çalışma dalına alındı.
