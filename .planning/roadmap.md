# AI Model Explorer — Roadmap

**Oluşturma:** 2026-06-09
**Kaynak:** `.planning/codebase/` (gsd-codebase-mapper analizi)

---

## Faz 1: Temel Sağlık & Bakım
*Düşük risk, yüksek etki — tüm fazların temeli.*

| # | Görev | Dosyalar | Süre |
|---|-------|----------|------|
| 1.1 | `except Exception: pass` → loguru logging | `app.py`, `core/cache_db.py`, `downloads/download_service.py` | 1g |
| 1.2 | Lazy import'ları module seviyesine taşı | `app.py` (~5 yer) | 0.5g |
| 1.3 | HTTP connection pooling (`requests.Session`) | Tüm provider'lar | 1g |
| 1.4 | Provider retry/backoff (`tenacity`) | `providers/hf_provider.py`, `providers/ollama_provider.py` | 1g |
| 1.5 | Error type hierarchy oluştur | `core/errors.py` (yeni) | 0.5g |

**Toplam:** ~4g · **Test:** Mevcut testler yeşil + yeni error path testleri

---

## Faz 2: Monoliti Kır — `app.py`
*En kritik teknik borç. 2466 satır → modüler yapı.*

| # | Görev | Hedef | Süre |
|---|-------|-------|------|
| 2.1 | Modal screen'leri ayır | `app/modals.py` (4 modal) | 1.5g |
| 2.2 | Widget'ları ayır | `app/widgets.py` (`SystemInfoWidget`) | 0.5g |
| 2.3 | Download lifecycle'ı delegate'e çıkar | `app/download_delegate.py` | 2g |
| 2.4 | Search orchestration'ı coordinator'a çıkar | `app/search_coordinator.py` | 2g |
| 2.5 | `AIModelViewer`'ı compose et | `app/app.py` (yeni, ~600 satır) | 2g |

**Toplam:** ~8g · **Risk:** YÜKSEK — regression riskine karşı Faz 3 ile paralel test yazılmalı

---

## Faz 3: Test Coverage Artırımı
*Hedef: %40 → %60 coverage gate.*

| # | Görev | Detay | Süre |
|---|-------|-------|------|
| 3.1 | Faz 2 çıktılarına unit test | Yeni modüller + mevcut `app.py` davranışı | 3g |
| 3.2 | Download service testleri | Worker loop, cancellation, subprocess | 1.5g |
| 3.3 | Ollama scraping mock testleri | HTML fixture + parse test | 1g |
| 3.4 | Error path coverage | Tüm `except` blokları test edilsin | 1.5g |
| 3.5 | Coverage gate güncelle | `fail_under = 40` → `60` | 0.5g |

**Toplam:** ~7.5g · **Not:** Faz 2 ile paralel ilerleyebilir

---

## Faz 4: Mimari İyileştirme
*Performans ve bakım kolaylığı.*

| # | Görev | Detay | Süre |
|---|-------|-------|------|
| 4.1 | Duplicate `estimate_model_size_gb` tekilleştir | `utils.py` sürümünü deprecate et | 0.5g |
| 4.2 | Provider aramalarını paralelleştir | `ThreadPoolExecutor` ile eşzamanlı sorgu | 1g |
| 4.3 | SQLite connection pooling | Connection-per-operation → module-level pool | 1g |
| 4.4 | `find_gpu_bandwidth` optimizasyonu | Linear scan → normalized lookup table | 0.5g |
| 4.5 | GPU bandwidth linear search optimize | Build trie at module load time | 0.5g |

**Toplam:** ~3.5g · **Bağımlılık:** Faz 2 tamamlanmış olmalı

---

## Faz 5: Provider Sağlamlaştırma
*Güvenlik ve sağlamlık.*

| # | Görev | Detay | Süre |
|---|-------|-------|------|
| 5.1 | Ollama scraping → resmi API | Ollama API dokümantasyonu araştır + implemente et | 2g |
| 5.2 | HF token env → stdin pipe | Güvenlik iyileştirmesi | 0.5g |
| 5.3 | `api_server.py` input validasyonu | 400 vs 500 hata yönetimi | 0.5g |
| 5.4 | `shell=True` kaldır (legacy) | `terminal_ui/app.py` | 0.5g |
| 5.5 | Hataları string → structured error tipleri | Tüm provider `search()` yöntemleri | 1g |

**Toplam:** ~4.5g · **Bağımlılık:** Faz 1 (error types), Faz 4 (paralel arama)

---

## Faz 6: Performans & UX
*Kullanıcıya yansıyan iyileştirmeler.*

| # | Görev | Detay | Süre |
|---|-------|-------|------|
| 6.1 | DataTable incremental update | `refresh_table()` rebuild → `update_cell()` | 1.5g |
| 6.2 | Search cancellation | Yeni arama eskiyi iptal etsin | 1g |
| 6.3 | Download multi-worker | `max_workers` ile paralel indirme | 1.5g |
| 6.4 | Offline/cache-first mod | Ağ yoksa önbellekten oku | 2g |
| 6.5 | UI donma azaltma | Uzun işlemlerde progress feedback | 1g |

**Toplam:** ~7g · **Bağımlılık:** Faz 2 + Faz 4

---

## Dependency Graph

```
Faz 1 (Temel Bakım)
   │
   ▼
Faz 2 (Monolit Kırma) ──► Faz 3 (Test Coverage) [paralel]
   │                           │
   ▼                           ▼
Faz 4 (Mimari İyileştirme) ◄──┘
   │
   ▼
Faz 5 (Provider Sağlamlaştırma)
   │
   ▼
Faz 6 (Performans & UX)
```

---

## Toplam Tahmini Süre

| Faz | Gün |
|-----|-----|
| Faz 1 — Temel Sağlık | 4g |
| Faz 2 — Monolit Kırma | 8g |
| Faz 3 — Test Coverage | 7.5g |
| Faz 4 — Mimari İyileştirme | 3.5g |
| Faz 5 — Provider Sağlamlaştırma | 4.5g |
| Faz 6 — Performans & UX | 7g |
| **Toplam** | **~34.5g** |

*Tam zamanlı tek geliştirici — tahmini 7 hafta.*
