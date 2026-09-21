"""
app.py
------
Streamlit Dashboard -- yalnızca görselleştirme katmanı.
Tüm iş mantığı (skorlama: scoring.py, ML: ml_model.py, aksiyon kuralları:
actions.py) başka modüllerde yaşar; bu dosya onları çağırıp ekrana çizer.
"""

import streamlit as st

from ml_model import AyrilisTahminlemeSistemi
from data.sample_data import gecmis_egitim_verisi, guncel_personel_verisi

# --- Sayfa Ayarları ---
st.set_page_config(page_title="İK Risk Dashboard", layout="wide", page_icon="📊")

st.title("🏃‍♂️ Personel Ayrılış (Turnover) Tahminleme Dashboard")
st.markdown("Makine öğrenmesi destekli risk analizi, otonom İK aksiyonları ve XAI raporlaması.")

# --- 1. Motoru başlat ve veriyi yükle ---
ik_sistemi = AyrilisTahminlemeSistemi()
gecmis_data = gecmis_egitim_verisi()
yeni_personel_data = guncel_personel_verisi()

# --- 2. Modeli eğit ve tahmin üret ---
with st.spinner("Makine Öğrenmesi modeli eğitiliyor..."):
    ik_sistemi.modeli_egit(gecmis_data)
    dashboard_sonucu = ik_sistemi.risk_tahmin_et_ve_aksiyon_belirle(yeni_personel_data)
    xai_raporu = ik_sistemi.aciklanabilir_yapay_zeka_raporu()

# --- 3. Ekran çizimi ---
RENK_HARITASI = {
    "Kritik": "#ffcccc",
    "Yüksek": "#ffebcc",
    "Dikkat": "#e6f2ff",
    "Düşük": "#e6ffe6",
}

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📋 Yönetici Risk Raporu ve Otonom Aksiyonlar")
    st.dataframe(
        dashboard_sonucu.style.applymap(
            lambda val: f"background-color: {RENK_HARITASI.get(val, '')}",
            subset=["Risk Sınıfı"],
        ),
        use_container_width=True,
    )

    st.subheader("📈 Personel Ayrılış Risk Dağılımı")
    st.bar_chart(data=dashboard_sonucu, x="sicil_no", y="final_ml_risk_skoru", color="#D9534F")

with col2:
    st.subheader("🧠 Model Karar Ağırlıkları (XAI)")
    st.info("Algoritma risk hesaplaması yaparken aşağıdaki metrikleri önceliklendirdi.")
    st.dataframe(xai_raporu, use_container_width=True)

    st.metric(label="Değerlendirilen Personel", value=len(yeni_personel_data))
    kritik_sayisi = len(dashboard_sonucu[dashboard_sonucu["Risk Sınıfı"] == "Kritik"])
    st.metric(
        label="Kritik Riskli Personel",
        value=kritik_sayisi,
        delta="-Acil Aksiyon Bekliyor",
        delta_color="inverse",
    )
