"""
data/sample_data.py
--------------------
Demo amaçlı simülasyon verileri. Gerçek kullanımda bu veriler İK sisteminden
(HRIS, DPYS, anket platformu vb.) çekilmeli ve bu dosyanın yerini almalıdır.
"""

import pandas as pd


def gecmis_egitim_verisi() -> pd.DataFrame:
    """Model eğitimi için geçmiş (etiketli) veri: istifa_durumu bilinen çalışanlar."""
    return pd.DataFrame(
        {
            "sicil_no": [101, 102, 103, 104, 105],
            "unvan": ["B1 Teknisyen", "Mühendis", "Uzman", "Mühendis", "B1 Teknisyen"],
            "ninebox_star_mi": [True, False, False, True, False],
            "devamsizlik_endeksi": [20, 60, 0, 10, 80],
            "puantaj_disiplini": [10, 115, 0, 10, 90],
            "dpys_performans_riski": [10, 50, 0, 0, 70],
            "izin_davranisi": [0, 40, 0, 20, 40],
            "ona_dijital_risk_skoru": [80, 20, 0, 90, 20],
            "anket_ortalamasi": [4.5, 2.1, 4.8, 3.5, 1.5],
            "demografik_ham_risk": [20, 60, 10, 40, 70],
            "makro_ham_risk": [80, 20, 10, 90, 30],
            "istifa_durumu": [1, 1, 0, 0, 1],
        }
    )


def guncel_personel_verisi() -> pd.DataFrame:
    """Tahmin yapılacak, henüz etiketlenmemiş güncel çalışan verisi."""
    return pd.DataFrame(
        {
            "sicil_no": [2231, 1452, 3390, 1187],
            "unvan": ["Kıdemli Müh.", "Kalite Uzmanı", "B1 Teknisyen", "Sistem Destek"],
            "ninebox_star_mi": [True, False, False, False],
            "devamsizlik_endeksi": [10, 60, 20, 0],
            "puantaj_disiplini": [10, 70, 20, 0],
            "dpys_performans_riski": [0, 50, 30, 20],
            "izin_davranisi": [0, 40, 0, 0],
            "ona_dijital_risk_skoru": [85, 20, 80, 10],
            "anket_ortalamasi": [4.0, 2.5, 3.8, 4.2],
            "demografik_ham_risk": [30, 80, 20, 40],
            "makro_ham_risk": [90, 20, 10, 10],
        }
    )
