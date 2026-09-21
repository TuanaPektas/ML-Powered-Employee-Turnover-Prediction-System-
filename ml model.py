"""
ml_model.py
-----------
Makine öğrenmesi katmanı: RandomForest ile ayrılış olasılığı tahmini ve
Açıklanabilir Yapay Zeka (XAI) raporu.

Uzman-sistem skorlarını (scoring.py) feature olarak kullanır, risk sınıfı ve
aksiyon kararını actions.py'ye devreder. Bu modül görselleştirme içermez.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from scoring import uzman_sistem_puanlarini_hesapla
from actions import otonom_aksiyon_belirle, ona_riski_yuksek_mi

OZELLIKLER = [
    "davranis_ona_skoru",
    "anket_skoru",
    "demografik_skor",
    "makroekonomik_skor",
]


class AyrilisTahminlemeSistemi:
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
        )
        self.model_egitildi_mi = False
        self.ozellikler = OZELLIKLER

    def modeli_egit(self, egitim_verisi: pd.DataFrame) -> None:
        """Geçmiş (etiketli, istifa_durumu kolonu olan) veriyle modeli eğitir."""
        islenmis_veri = uzman_sistem_puanlarini_hesapla(egitim_verisi)
        X = islenmis_veri[self.ozellikler]
        y = islenmis_veri["istifa_durumu"]

        self.rf_model.fit(X, y)
        self.model_egitildi_mi = True

    def risk_tahmin_et_ve_aksiyon_belirle(self, yeni_calisanlar: pd.DataFrame) -> pd.DataFrame:
        """
        Yeni çalışanlar için ML tabanlı risk skoru (0-100) ve buna bağlı
        otonom İK aksiyonunu üretir. Sınıf/aksiyon kararı actions.py'den gelir.
        """
        if not self.model_egitildi_mi:
            raise RuntimeError("Model henüz eğitilmedi! Önce modeli_egit() fonksiyonunu çağırın.")

        islenmis_veri = uzman_sistem_puanlarini_hesapla(yeni_calisanlar)
        X_yeni = islenmis_veri[self.ozellikler]

        risk_olasiliklari = self.rf_model.predict_proba(X_yeni)[:, 1] * 100
        islenmis_veri["final_ml_risk_skoru"] = np.round(risk_olasiliklari, 2)

        aksiyon_listesi = []
        for _, row in islenmis_veri.iterrows():
            sonuc = otonom_aksiyon_belirle(
                skor=row["final_ml_risk_skoru"],
                star_mi=bool(row.get("ninebox_star_mi", False)),
                ona_riski_yuksek_mi=ona_riski_yuksek_mi(row["ona_dijital_risk_skoru"]),
            )
            aksiyon_listesi.append({"Risk Sınıfı": sonuc.sinif, "Otonom Aksiyon": sonuc.aksiyon})

        aksiyon_df = pd.DataFrame(aksiyon_listesi, index=islenmis_veri.index)
        sonuc_df = pd.concat(
            [islenmis_veri[["sicil_no", "unvan", "final_ml_risk_skoru"]], aksiyon_df],
            axis=1,
        )
        return sonuc_df

    def aciklanabilir_yapay_zeka_raporu(self) -> pd.DataFrame:
        """Modelin karar ağırlıklarını (feature importance) DataFrame olarak döner."""
        if not self.model_egitildi_mi:
            raise RuntimeError("Model henüz eğitilmedi! Önce modeli_egit() fonksiyonunu çağırın.")

        onem_dereceleri = pd.DataFrame(
            {
                "Metrik": self.ozellikler,
                "Etki_Yuzdesi": np.round(self.rf_model.feature_importances_ * 100, 2),
            }
        ).sort_values(by="Etki_Yuzdesi", ascending=False)
        return onem_dereceleri
