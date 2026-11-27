"""Sunum günü kontrolü: demoyu başlatır, üç ana sayfayı açar, hata olmadığını doğrular.

Kullanım (proje kökünden):
    python scripts/sunum_kontrol.py                    # yalnız kontrol
    python scripts/sunum_kontrol.py --ekran-goruntusu  # ayrıca 1280×720 yedek görüntüler: docs/sunum/yedek/

Ne yapar:
  1. `streamlit run app/streamlit_app.py`'yi ayrı bir portta (varsayılan 8502) başlatır; sağlık adresine yanıt bekler.
  2. Talep Tahmini, Anomali Tespiti ve Gerçek Etiketli Veri (LEAD) sayfalarını Playwright (Chromium) ile 1280×720
     pencerede açar. Her sayfada başlığın, en az bir grafiğin göründüğünü ve hiçbir hata öğesi
     (Streamlit istisnası ya da st.error kutusu) olmadığını denetler; yükleme süresini yazar.
  3. İstenirse her sayfanın varsayılan görünümünden iki 1280×720 görüntü kaydeder: sayfanın üstü ve ana grafik
     (demo çökerse slayta konur).
Hiçbir model eğitilmez; demo yalnız kayıtlı dosyaları okur. Gerekenler: `pip install -r requirements-dev.txt`
ve bir kez `python -m playwright install chromium`.
"""

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # yalnız `src` import'u için (verify betiği gibi)
from src.core.config import PROJECT_ROOT                                    # noqa: E402

PAGES = [   # (adres, beklenen başlık, ekran görüntüsü adı)
    ("talep-tahmini", "Talep Tahmini", "01_talep_tahmini.png"),
    ("anomali-tespiti", "Anomali Tespiti", "02_anomali_tespiti.png"),
    ("gercek-etiketli-veri", "Gerçek Etiketli Veri (LEAD)", "03_lead.png"),
]
ERROR_SELECTORS = '[data-testid="stException"], [data-testid="stAlertContentError"]'


def start_app(port):
    """Demoyu arka planda başlatır ve sağlık adresi yanıt verene kadar (en çok 90 s) bekler."""
    command = [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.headless", "true",
               "--server.address", "localhost", "--server.port", str(port)]
    process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(90):
        try:
            if urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=2).status == 200:
                return process
        except OSError:
            time.sleep(1)
    process.terminate()
    raise RuntimeError("Demo 90 saniyede başlamadı.")


def check_page(page, base, path, title, timeout_s):
    """Sayfayı açar; başlık + grafik görünene ve çalışma bitene kadar bekler. Çıktı: (süre s, hata metinleri)."""
    started = time.time()
    page.goto(f"{base}/{path}")
    page.get_by_role("heading", name=title).first.wait_for(timeout=timeout_s * 1000)
    page.locator(".js-plotly-plot").first.wait_for(timeout=timeout_s * 1000)
    # Streamlit çalışırken sağ üstte "Running" göstergesi vardır; bitmesini bekle.
    page.wait_for_function("!document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=timeout_s * 1000)
    page.wait_for_timeout(1500)                                   # grafiklerin son çizimi
    elapsed = time.time() - started
    errors = [element.inner_text()[:200] for element in page.locator(ERROR_SELECTORS).all()]
    return elapsed, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8502)
    parser.add_argument("--ekran-goruntusu", action="store_true", help="docs/sunum/yedek/ altına 1280×720 PNG kaydet")
    parser.add_argument("--timeout", type=int, default=120, help="sayfa başına en çok bekleme (s)")
    args = parser.parse_args()
    from playwright.sync_api import sync_playwright

    out_dir = PROJECT_ROOT / "docs" / "sunum" / "yedek"
    process = start_app(args.port)
    failures = 0
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            for path, title, image in PAGES:
                try:
                    elapsed, errors = check_page(page, f"http://localhost:{args.port}", path, title, args.timeout)
                except Exception as error:                          # zaman aşımı vb.: başarısız say
                    elapsed, errors = float("nan"), [f"{type(error).__name__}: {str(error)[:200]}"]
                status = "TAMAM" if not errors else "HATA"
                failures += bool(errors)
                print(f"{status:5s} {title:30s} {elapsed:5.1f} s" + ("" if not errors else f"  → {errors}"))
                if args.ekran_goruntusu and not errors:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    page.screenshot(path=str(out_dir / image))                      # sayfanın üstü
                    chart = out_dir / image.replace(".png", "_grafik.png")
                    page.locator(".js-plotly-plot").first.scroll_into_view_if_needed()
                    page.wait_for_timeout(800)
                    page.screenshot(path=str(chart))                                # ana grafik görünümde
                    print(f"      ekran görüntüleri: {(out_dir / image).relative_to(PROJECT_ROOT).as_posix()}, "
                          f"{chart.name}")
            browser.close()
    finally:
        process.terminate()
    print("Bütün sayfalar hatasız." if not failures else f"{failures} sayfada sorun var.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
