"""Open-Meteo (ERA5) saatlik hava verisini indirir ve veri setinin yerel saatine hizalar.

Kaynak: Open-Meteo Historical Weather API (archive-api.open-meteo.com), ERA5
yeniden analizi. Lisans: CC BY 4.0 — atıf gerekir (README §2).

Saat dilimi: API'den UTC istenir. UCI veri seti Europe/Paris YEREL saatindedir
(yaz saatli): her ilkbahar geçişinde var olmayan 02:xx saati 03:xx'in kopyasıdır,
sonbaharda tekrar eden 02:xx saati yalnız bir kez geçer (deney günlüğü, Aşama 3b).
"""

import argparse
import json
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from src.data.load import PROJECT_ROOT, load_config
from src.utils.console import setup_console
from src.utils.results_io import resolve_path


def download_weather(settings, path):
    """API'den saatlik hava verisini UTC olarak indirip CSV'ye yazar.

    Girdi: config "weather" bölümü, hedef dosya yolu.
    Çıktı: [saat, time_utc + değişkenler] DataFrame (dosyaya da yazılır).
    """
    query = urlencode({
        "latitude": settings["latitude"],
        "longitude": settings["longitude"],
        "start_date": settings["start_date"],
        "end_date": settings["end_date"],
        "hourly": ",".join(settings["variables"]),
        "models": settings["model"],
        "timezone": "GMT",                      # açıkça UTC iste
    })
    with urlopen(f"{settings['url']}?{query}", timeout=120) as response:
        payload = json.load(response)
    if payload.get("utc_offset_seconds") != 0:
        raise RuntimeError("API UTC dışında bir saat dilimi döndürdü.")
    frame = pd.DataFrame(payload["hourly"]).rename(columns={"time": "time_utc"})
    print(f"ERA5 hücresi: enlem {payload['latitude']}, boylam {payload['longitude']}, "
          f"yükseklik {payload['elevation']} m")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def load_weather(path):
    """Ham CSV'yi UTC (tz-aware) indeksli DataFrame olarak okur."""
    frame = pd.read_csv(path)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("time_utc")), name="time_utc").tz_localize("UTC")
    return frame


def local_to_utc(local_index, timezone):
    """Veri setinin saf (tz'siz) yerel saat indeksini UTC'ye çevirir.

    Girdi: saatlik yerel DatetimeIndex (tz'siz), saat dilimi adı.
    Çıktı: aynı uzunlukta UTC DatetimeIndex.
    Kurallar:
    - İlkbahar: yerelde var olmayan 02:00, 03:00'e kaydırılır (shift_forward);
      veri setinde bu saat zaten 03:xx'in kopyası olduğundan hava da 03:00'ün havası olur.
    - Sonbahar: iki kez yaşanan 02:00 yaz saati (ilk geçiş) kabul edilir; veri
      setinde bu saat bir kez geçtiği için UTC'de bir saat atlanır (yılda 1 saat).
    """
    ambiguous_as_dst = np.ones(len(local_index), dtype=bool)
    return local_index.tz_localize(timezone, ambiguous=ambiguous_as_dst,
                                   nonexistent="shift_forward").tz_convert("UTC")


def align_weather(local_index, weather_utc, timezone):
    """UTC hava tablosunu veri setinin yerel saat indeksine hizalar.

    Girdi: yerel saatlik indeks, load_weather çıktısı, saat dilimi.
    Çıktı: yerel indeksli hava tablosu (eşleşmeyen saatler NaN).
    """
    aligned = weather_utc.reindex(local_to_utc(local_index, timezone))
    aligned.index = local_index
    return aligned


def add_weather_features(df, weather_utc, settings, weather_columns):
    """Hava değişkenlerini ve derece-saatleri saatlik tabloya ekler (kopya döner).

    Girdi: saatlik tablo (yerel saat), UTC hava tablosu, config "weather"
    bölümü, eklenecek sütun adları (config features.weather).
    Çıktı: hava sütunları eklenmiş yeni DataFrame.
    Neden: Isıtma derece-saati max(0, 18 − T) soğuk havada artan ısıtma
    ihtiyacını, soğutma derece-saati max(0, T − 24) sıcakta klima ihtiyacını
    doğrusal hâle getirir. Değerler yalnız o saate aittir (gelecek bilgisi yok).
    """
    aligned = align_weather(df.index, weather_utc, settings["local_timezone"])
    result = df.copy()
    for column in settings["variables"]:
        result[column] = aligned[column].to_numpy(dtype=np.float64)
    temperature = result["temperature_2m"]
    result["heating_degree_hours"] = (settings["heating_base_c"] - temperature).clip(lower=0)
    result["cooling_degree_hours"] = (temperature - settings["cooling_base_c"]).clip(lower=0)
    missing = result[weather_columns].isna().sum()
    if missing.any():
        # Doldurma yapılmaz (bfill yasak); eksik varsa eğitim durdurulur.
        raise ValueError(f"Hava verisinde eksik saat var:\n{missing[missing > 0]}")
    return result


def load_weather_frame(df, config):
    """Saatlik tabloya config'teki hava dosyasından öznitelikleri ekler."""
    settings = config["weather"]
    weather = load_weather(resolve_path(PROJECT_ROOT, settings["file"]))
    return add_weather_features(df, weather, settings, config["features"]["weather"])


def main():
    """Dosya yoksa (veya --force) indirir; eksik değer ve hizalama özetini yazdırır."""
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Dosya varsa da yeniden indir")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    settings = config["weather"]
    path = resolve_path(PROJECT_ROOT, settings["file"])
    if args.force or not path.exists():
        download_weather(settings, path)
    weather = load_weather(path)
    print(f"Ham hava: {len(weather)} saat, {weather.index.min()} → {weather.index.max()} (UTC)")
    print("Ham dosyada eksik değer:\n" + weather.isna().sum().to_string())

    df = pd.read_parquet(resolve_path(PROJECT_ROOT, config["paths"]["processed_file"]))
    aligned = align_weather(df.index, weather, settings["local_timezone"])
    print("Hizalama sonrası eksik saat (veri seti ekseninde):\n" + aligned.isna().sum().to_string())
    for local in ("2008-01-15 12:00", "2008-07-15 12:00", "2008-03-30 02:00", "2008-10-26 02:00"):
        utc = local_to_utc(pd.DatetimeIndex([local]), settings["local_timezone"])[0]
        print(f"yerel {local} → UTC {utc:%Y-%m-%d %H:%M}, T = {aligned.loc[local, 'temperature_2m']} °C")


if __name__ == "__main__":
    main()
