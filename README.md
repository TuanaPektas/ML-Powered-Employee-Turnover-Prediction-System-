# ML-Powered-Employee-Turnover-Prediction-System-
Personnel departure based on machine learning support, expert-system scoring  Risk forecasting and autonomous HR action engine. An interactive streamlit  It is presented as an administrator dashboard.

# 🏃‍♂️ Personel Ayrılış (Turnover) Tahminleme Sistemi

Makine öğrenmesi destekli, uzman-sistem skorlamasına dayanan personel ayrılış
riski tahminleme ve otonom İK aksiyon motoru. Streamlit ile interaktif bir
yönetici dashboard'u olarak sunulur.

## Mimari

Proje, matematik/iş mantığı ile arayüzü net biçimde ayıran katmanlı bir
yapıya sahiptir:

```
├── scoring.py          # Uzman sistem formülleri (A, B, C, D bileşenleri)
├── actions.py          # Risk sınıflandırması + otonom aksiyon kuralları
├── ml_model.py          # RandomForest eğitim/tahmin + XAI raporu
├── data/
│   └── sample_data.py   # Demo simülasyon verisi
├── app.py                # Streamlit dashboard (yalnızca görselleştirme)
├── requirements.txt
└── README.md
```

Her katman bağımsız olarak test edilebilir; örneğin `scoring.py` hiçbir ML
veya UI bağımlılığı içermez.

## Skorlama Modeli (v2)

`Final Risk Skoru = A×0.50 + B×0.20 + C×0.15 + D×0.15`

| Bileşen | Açıklama | Ağırlık |
|---|---|---|
| A | Davranış ve ONA Score (devamsızlık, puantaj, performans, izin, organizasyon/ONA) | %50 |
| B | Anket Score (memnuniyet anketi, 1-5 ölçek) | %20 |
| C | Demografik Score (kıdem, yaş kuşağı, kariyer durgunluğu) | %15 |
| D | Makroekonomik Score (rakip ilan yoğunluğu, sektörel açık, ücret endeksi) | %15 |

Tüm alt bileşenler 0-100 aralığına normalize edilir (`N(x, max) = (x/max)×100`),
böylece ağırlıklı toplam matematiksel olarak 0-100 aralığında garanti kalır.

Üretimde bu kural-tabanlı skor yerine, aynı dört bileşen feature olarak bir
`RandomForestClassifier`'a verilir ve gerçek geçmiş ayrılış verisiyle eğitilen
model, olasılıksal bir risk skoru üretir.

### Risk Sınıflandırması

| Aralık | Sınıf | Otonom Aksiyon |
|---|---|---|
| 0–29 | Düşük | Periyodik izleme |
| 30–49 | Dikkat | Yöneticiye 15 dk "Kahve Sohbeti" ataması |
| 50–69 | Yüksek | Star ise İç İşe Alım, değilse gelişim planı |
| 70–100 | Kritik | ONA/Şef istifası kaynaklıysa acil "Görüşme Yap", değilse Sağlık/Disiplin süreci |

## Kurulum

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Çalıştırma

```bash
streamlit run app.py
```

## Notlar
- `data/sample_data.py` demo amaçlıdır; gerçek kullanımda HRIS/DPYS/anket
  sistemlerinden veri çekecek bir veri katmanıyla değiştirilmelidir.
