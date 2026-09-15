# Kararlar (ADR-lite)

Bir şeyin **neden** böyle olduğunu ararken önce buraya bak. Yeni kalıcı
karar alınca buraya ekle. Biçim: **Karar** / **Gerekçe** / **Sonuç**.

---

## K-1 — GitHub Actions bu hesapta çalışmıyor; zamanlı iş süreç ya da sunucu tarafında kurulur

**Tarih:** 2026-09-15 · **Kaynak:** kullanıcı kararı; `STORK-faz-1`'in 21 deposunda
aynı gün uygulandı, burası kişisel hesaptaki karşılığıdır.

**Karar.** Bu depoya `schedule:` tetikleyicili bir GitHub Actions workflow'u
**eklenmez**. Tekrarlayan her iş ya sürecin kendi zamanlayıcısına ya da
sunucudaki cron'a / systemd timer'ına kurulur.

**Gerekçe — varsayılmadı, ÖLÇÜLDÜ (2026-09-15, GitHub API).** `Ozgurisikdamar`
hesabında GitHub Actions **bugüne kadar hiç koşmadı.** Workflow'u olan iki depo:

| depo | koşum | ilk koşum | sonuç |
|---|---|---|---|
| `portfolyo` | 25 | 2026-05-29 — `lighthouse.yml`'i **ekleyen** commit | 25/25 `startup_failure` |
| `ty_discovery` | 30 | 2026-09-04 | 30/30 `startup_failure` |

Toplam **55/55 `startup_failure`**, sıfır başarılı koşum. Hepsi
`created_at == run_started_at == updated_at` ile **anında** düşüyor (sıfır saniye
derleme) ve `path` alanı gerçek workflow yolu yerine `BuildFailed` yazıyor — yani
koşum workflow dosyasını okumaya **hiç ulaşmıyor**. Sebep dosyada değil, hesap
düzeyinde (Actions kapalı / faturalandırma / harcama limiti).

⚠️ **`STORK-faz-1` kuralının bıraktığı kaçış kapısı BURADA YOK.** Org metni
"`workflow_dispatch` durur — elle tetikleme her zaman mümkün" der; bu hesap için
**yanlış**. `ty_discovery/live-probe.yml` 2026-09-05'te elle tetiklendi
(`event: workflow_dispatch`) ve o koşum da `startup_failure` döndü. Bu hesapta
**hiçbir tetikleyici** çalışmıyor — elle olan dahil.

⚠️ **Arıza biçimi de org'unkinden farklı ve daha yanıltıcı.** Org'da koşum
`queued`'da sessizce bekler, kırmızı yanmaz. Burada **kırmızı yanar ama iş log'u
yoktur** → bozuk bir YAML gibi görünür ve insanı saatlerce YAML hata ayıklamaya
iter. YAML'ın suçu yok; koşum oraya hiç varmadı.

**Sonuç.**

- Bu depoda `schedule:` sayısı **0** (2026-09-15'te ölçüldü); kural ileriye dönüktür.
- Bu depodaki tahmin betikleri (`lstm.py`, `ts_prophet.py`, `main.py`) **elle**
  koşturulur; zamanlanmış bir iş yoktur.
- Düzenli tahmin gerekirse yeri: betiği koşturan makinenin cron'u ya da sürecin
  kendi döngüsü — Actions değil.
- Kontrol **yereldir**, çünkü bu kuralın bekçisi bir workflow **olamaz** (o da
  koşmazdı): `grep -rE '^[[:space:]]*schedule:' .github/workflows/` → 0 satır.
- Actions yeniden çalışır hale gelirse bu kayıt güncellenir. O güne kadar bu
  depoda **"CI yeşil" diye bir kanıt yoktur**; tek gerçek kapı yerel koşumdur.

---

## K-2 — Bu depo `stork-contracts` sözleşmesine katılmaz

**Tarih:** 2026-09-15

**Karar.** STORK ekosisteminin depolar arası sözleşme deposu
(`STORK-faz-1/stork-contracts`) bu depoyu **bağlamaz**.

**Gerekçe.** Bu depo STORK ekosisteminin parçası değildir; ortak şema, event ya da fixture kullanmaz.

**Sonuç.** Sözleşme deposunu okumak gerekmez; bir gün bağ kurulursa karar
burada tam metin yazılır (işaretçi değil).
