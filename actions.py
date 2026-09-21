"""
actions.py
----------
Risk skorunu (0-100) risk sınıfına ve otonom İK aksiyonuna çeviren kural
motoru. v2 dokümanındaki sınır tanımına göre her aralık kendi üst sınırını
İÇERMEZ (70+ hariç):

    0  - 29  : Düşük
    30 - 49  : Dikkat
    50 - 69  : Yüksek
    70 - 100 : Kritik
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskSonucu:
    sinif: str
    aksiyon: str


def risk_sinifi_belirle(skor: float) -> str:
    if 70 <= skor <= 100:
        return "Kritik"
    elif 50 <= skor < 70:
        return "Yüksek"
    elif 30 <= skor < 50:
        return "Dikkat"
    else:
        return "Düşük"


def otonom_aksiyon_belirle(
    skor: float,
    star_mi: bool = False,
    ona_riski_yuksek_mi: bool = False,
) -> RiskSonucu:
    """
    Risk skoru ve bağlamsal bayraklara (star yetenek, ONA/dalga etkisi riski)
    göre risk sınıfını ve tetiklenecek otonom İK aksiyonunu döndürür.
    """
    sinif = risk_sinifi_belirle(skor)

    if sinif == "Kritik":
        aksiyon = (
            "ACİL MÜDAHALE: ONA Dalga Etkisi. 1:1 'Stay Interview' başlatıldı."
            if ona_riski_yuksek_mi
            else "ACİL MÜDAHALE: Sağlık/Disiplin süreci incelemesi."
        )
    elif sinif == "Yüksek":
        aksiyon = (
            "İÇ İŞE ALIM (Internal Mobility): Rotasyon teklifi hazırla."
            if star_mi
            else "Gelişim planı / Eğitim ataması yap."
        )
    elif sinif == "Dikkat":
        aksiyon = "[AKSİYON GEREKLİ] Yöneticiye 15 dk 'Kahve Sohbeti' takvimi atandı."
    else:
        aksiyon = "Sadece periyodik izleme. Aksiyon yok."

    return RiskSonucu(sinif=sinif, aksiyon=aksiyon)


def ona_riski_yuksek_mi(ona_dijital_risk_skoru: float, esik: float = 70) -> bool:
    """ONA dijital risk skoru eşik değerin üzerindeyse True döner (varsayılan eşik: 70)."""
    return ona_dijital_risk_skoru >= esik
