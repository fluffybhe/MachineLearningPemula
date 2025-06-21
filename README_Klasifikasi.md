
# 🎯 Submission Akhir - Klasifikasi BMLP
**Nama:** Febhe Maulita May Pramasta  
**Proyek:** Klasifikasi Data Menggunakan Machine Learning  
**Program:** Laskar AI 2025

---

## **1. Import Library**
Pada tahap ini, kita mengimpor beberapa pustaka Python yang dibutuhkan untuk analisis data dan pembuatan model klasifikasi.

> ⚠️ Catatan: Terdapat warning dari pandas terkait versi `bottleneck` yang lebih rendah dari yang direkomendasikan.

---

## **2. Memuat Dataset dari Hasil Clustering**
Dataset hasil clustering dimuat ke dalam DataFrame untuk dilakukan klasifikasi lebih lanjut.

- Jumlah baris: 2205
- Jumlah kolom: 31

**Contoh Kolom:**
- Education
- Marital_Status
- Income
- Recency
- MntWines, MntFruits, MntMeatProducts, ...
- Cluster (target klasifikasi)

---

## **3. Data Splitting**
Dataset dibagi menjadi:
- **Training set**: 1323 data
- **Testing set**: 882 data

Target klasifikasi: `Cluster` (nilai 0, 1, atau 2)

---

## **4. Membangun Model Klasifikasi**

### a. Pelatihan Model
Model klasifikasi dilatih menggunakan algoritma:
- Decision Tree
- Naive Bayes

> Model training selesai.

### b. Evaluasi Model

#### Decision Tree Classifier
- Confusion Matrix:
  ```
  [[475   9   0]
   [ 19 314   1]
   [  1   5  58]]
  ```
- Accuracy: **96.03%**

#### Naive Bayes Classifier
- Confusion Matrix:
  ```
  [[469  12   3]
   [ 18 257  59]
   [  0   0  64]]
  ```
- Accuracy: **89.57%**

### c. Perbandingan Model

| Model               | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Decision Tree (DT) | 0.9603   | 0.9604    | 0.9603 | 0.9602   |
| Naive Bayes (NB)   | 0.8957   | 0.9271    | 0.8957 | 0.9018   |

---

## **5. Analisis Hasil Evaluasi**

- **Decision Tree** menunjukkan performa sangat baik dengan F1-Score 96%.
- **Naive Bayes** juga cukup andal meskipun F1-Score sedikit lebih rendah di 90%.
- **Rekomendasi**:
  - Tambahkan data latih
  - Pertajam segmentasi pelanggan untuk hasil klasifikasi yang lebih akurat

---

> 📘 *Dokumen ini dihasilkan dari Jupyter Notebook menggunakan konversi otomatis ke format Markdown. Untuk eksplorasi lebih lanjut, silakan akses notebook aslinya.*
