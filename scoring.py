"""
scoring.py
----------
Nihai Ayrılış Tahminlemesi (Turnover Prediction) - Uzman Sistem Skorlama Katmanı.

Bu modül SADECE matematiksel formülleri içerir (v2 - Matematiksel Olarak
Düzeltilmiş versiyon). Makine öğrenmesi veya görselleştirme ile ilgisi yoktur,
bu sayede bağımsız olarak unit test edilebilir.

Referans: Nihai Ayrılış Tahminlemesi Sistemi v2 dokümantasyonu.
"""

import pandas as pd


# --- Alt bileşenlerin maksimum ham puanları (normalizasyon için) ---
MAX_DEVAMSIZLIK = 80
MAX_PUANTAJ = 100  # cap uygulanır
MAX_PERFORMANS = 70
MAX_IZIN = 40
MAX_ONA = 100

# --- A bileşeni alt ağırlıkları (toplamı 1.00) ---
AGIRLIK_DEVAMSIZLIK = 0.20
AGIRLIK_PUANTAJ = 0.20
AGIRLIK_PERFORMANS = 0.20
AGIRLIK_IZIN = 0.10
AGIRLIK_ONA = 0.30

# --- Ana bileşen ağırlıkları (Final Risk Skoru, toplamı 1.00) ---
AGIRLIK_A = 0.50  # Davranış ve ONA Score
AGIRLIK_B = 0.20  # Anket Score
AGIRLIK_C = 0.15  # Demografik Score
AGIRLIK_D = 0.15  # Makroekonomik Score


def normalize(x, maks):
    """N(x, max) = (x / max) * 100 -- 0-100 aralığına normalize eder."""
    return (x / maks) * 100


def hesapla_a_davranis_ona(df: pd.DataFrame) -> pd.Series:
    """
    A. Davranış ve ONA Score (Etki: %50)

    Beklenen kolonlar:
        devamsizlik_endeksi, puantaj_disiplini, dpys_performans_riski,
        izin_davranisi, ona_dijital_risk_skoru
    """
    puantaj_capli = df["puantaj_disiplini"].clip(upper=MAX_PUANTAJ)

    a = (
        AGIRLIK_DEVAMSIZLIK * normalize(df["devamsizlik_endeksi"], MAX_DEVAMSIZLIK)
        + AGIRLIK_PUANTAJ * normalize(puantaj_capli, MAX_PUANTAJ)
        + AGIRLIK_PERFORMANS * normalize(df["dpys_performans_riski"], MAX_PERFORMANS)
        + AGIRLIK_IZIN * normalize(df["izin_davranisi"], MAX_IZIN)
        + AGIRLIK_ONA * normalize(df["ona_dijital_risk_skoru"], MAX_ONA)
    )
    return a


def hesapla_b_anket(df: pd.DataFrame) -> pd.Series:
    """
    B. Anket Score (Etki: %20)
    B = ((5 - Anket_Ortalaması) / 4) * 100
    Anket ortalaması 1-5 ölçek; risk memnuniyetle ters orantılı.
    """
    return ((5 - df["anket_ortalamasi"]) / 4) * 100


def hesapla_c_demografik(df: pd.DataFrame) -> pd.Series:
    """
    C. Demografik Score (Etki: %15)
    Alt puanlar (Kıdem 40 + Yaş 20 + Kariyer Durgunluğu 40) zaten 100'e
    tamamlandığı için ek normalizasyon GEREKMEZ; ham risk doğrudan kullanılır.
    """
    return df["demografik_ham_risk"].astype(float)


def hesapla_d_makroekonomik(df: pd.DataFrame) -> pd.Series:
    """
    D. Makroekonomik Score (Etki: %15)
    Alt puanlar (Rakip İlan 50 + Sektörel Açık 30 + Ücret Endeksi 20) zaten
    100'e tamamlandığı için ek normalizasyon GEREKMEZ.
    """
    return df["makro_ham_risk"].astype(float)


def uzman_sistem_puanlarini_hesapla(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ham metriklerden A, B, C, D bileşenlerini ve (opsiyonel) kural-tabanlı
    Final Risk Skorunu üretir. ML modeli bu dörtlüyü feature olarak kullanır.
    """
    sonuc = df.copy()
    sonuc["davranis_ona_skoru"] = hesapla_a_davranis_ona(sonuc)
    sonuc["anket_skoru"] = hesapla_b_anket(sonuc)
    sonuc["demografik_skor"] = hesapla_c_demografik(sonuc)
    sonuc["makroekonomik_skor"] = hesapla_d_makroekonomik(sonuc)
    return sonuc


def kural_tabanli_final_skor(df: pd.DataFrame) -> pd.Series:
    """
    Saf uzman-sistem (ML'siz) Final Risk Skoru -- v2 dokümanındaki ana formül.
    Final Risk Skoru = A*0.50 + B*0.20 + C*0.15 + D*0.15
    """
    hesaplanmis = uzman_sistem_puanlarini_hesapla(df)
    return (
        AGIRLIK_A * hesaplanmis["davranis_ona_skoru"]
        + AGIRLIK_B * hesaplanmis["anket_skoru"]
        + AGIRLIK_C * hesaplanmis["demografik_skor"]
        + AGIRLIK_D * hesaplanmis["makroekonomik_skor"]
    )
