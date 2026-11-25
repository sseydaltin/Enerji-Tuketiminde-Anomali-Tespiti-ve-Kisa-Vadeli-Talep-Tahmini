"""Konsol çıktısının kodlamasını (encoding) ayarlayan yardımcı fonksiyon."""

import sys


def setup_console() -> None:
    """Standart çıktı (stdout) ve hata çıktısını (stderr) UTF-8'e ayarlar.

    Girdi: Yok.
    Çıktı: Yok (sys.stdout ve sys.stderr'in kodlamasını değiştirir).

    Neden: Windows konsolu varsayılan olarak cp1252 kodlamasını kullanır ve
    bu kodlama "İ", "ş", "ğ" gibi Türkçe harfleri içermez. Betiklerdeki
    `print("SAATLİK VERİ ÖZETİ")` gibi satırlar bu yüzden
    UnicodeEncodeError ile çöküyordu. Her betiğin main() fonksiyonunun
    başında çağrılır; böylece kullanıcının PYTHONIOENCODING ayarlamasına
    gerek kalmaz.
    """
    for stream in (sys.stdout, sys.stderr):
        # Bazı ortamlar (ör. notebook) akışı reconfigure'ı olmayan bir
        # nesneyle değiştirir; o durumda dokunmadan geçiyoruz.
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
