# 🧅 Sistem Sortasi Mutu Bawang Merah berbasis AI

Sistem klasifikasi kualitas/mutu bawang merah menggunakan Computer Vision & Machine Learning, membandingkan dua pendekatan: **CNN (Deep Learning)** dan **SVM + GLCM (Feature Extraction klasik)** — dilengkapi aplikasi desktop untuk demo interaktif.

---

## 🎯 Tujuan Proyek

Mengotomatisasi proses sortasi mutu bawang merah yang biasanya dilakukan manual, dengan mengklasifikasikan gambar bawang merah ke dalam tiga kelas:

| Kelas | Keterangan |
|---|---|
| **Mutu I** | Kualitas baik |
| **Mutu II** | Kualitas sedang |
| **Tidak Layak** | Kualitas rendah / tidak layak jual |

---

## 🧠 Pendekatan Machine Learning

Proyek ini membandingkan **dua model klasifikasi** untuk melihat metode mana yang lebih akurat dan efisien:

1. **CNN (Convolutional Neural Network)**
   Deep learning langsung dari citra RGB, dilatih menggunakan TensorFlow/Keras dengan augmentasi data dan anti-overfitting (EarlyStopping).

2. **SVM + GLCM (Gray-Level Co-occurrence Matrix)**
   Pendekatan klasik: fitur tekstur (contrast, correlation, energy, homogeneity) diekstraksi dari citra grayscale menggunakan GLCM, lalu diklasifikasikan dengan Support Vector Machine.

Kedua model dievaluasi berdampingan (akurasi, confusion matrix, classification report, ROC curve) untuk menentukan pendekatan terbaik.

---

## 📊 Dataset

- Sumber: [Roboflow – Machine Learning Visual Texture](https://universe.roboflow.com/saitama-pasop/machine-learning-visual-texture-0eig3) (lisensi CC BY 4.0)
- Total **3.594 gambar**, format folder (train/valid/test)
- Pra-pemrosesan: auto-orientation, resize ke 224×224
- Augmentasi: rotasi 90°, rotasi acak ±28°, salt-and-pepper noise

---

## 🖥️ Aplikasi Desktop (Demo)

Proyek ini dilengkapi aplikasi GUI (Tkinter) untuk mendemonstrasikan model secara langsung:

- Upload gambar bawang merah → sistem menampilkan hasil klasifikasi dari **kedua model** (CNN & SVM) secara berdampingan
- Modal detail analisis: menampilkan gambar asli, grayscale, dan hasil ekstraksi fitur GLCM
- **`launcher.py`** — installer otomatis yang mengecek dan memasang seluruh dependency (`numpy`, `opencv-python`, `tensorflow`, `scikit-learn`, `scikit-image`, dll.) sebelum menjalankan aplikasi utama, lengkap dengan logging error ke `app_error.log`

---

## 📁 Struktur Proyek

```
├── Bawang_Merahipynb.ipynb              # Notebook: preprocessing, training CNN & SVM, evaluasi
├── app.py                                # Aplikasi GUI utama (Tkinter) untuk klasifikasi
├── launcher.py                           # Installer & pengecekan dependensi otomatis
├── jalankan.bat                          # Shortcut menjalankan aplikasi di Windows
├── Model_CNN_Bawang_Biner.h5             # Model CNN terlatih
├── Model_SVM_GLCM_Biner.pkl              # Model SVM+GLCM terlatih
├── Machine Learning Visual Texture..v3i.folder/   # Dataset (train/valid/test)
└── FODLER GAMBAR/                        # Contoh screenshot hasil aplikasi
```

---

## 🛠️ Teknologi

- **Python** — TensorFlow/Keras (CNN), Scikit-learn (SVM), Scikit-image (GLCM)
- **OpenCV & Pillow** — pemrosesan citra
- **Tkinter** — antarmuka aplikasi desktop
- **Google Colab** — environment training model (Notebook)

---

## 🚀 Menjalankan Aplikasi

### Cara termudah (Windows)
```
1. Jalankan launcher.py atau klik jalankan.bat
2. Launcher otomatis mengecek & memasang dependensi yang diperlukan
3. Aplikasi utama (app.py) akan terbuka otomatis
```

### Manual
```bash
pip install numpy opencv-python Pillow scikit-learn scikit-image joblib tensorflow
python app.py
```

> Catatan: training model (di notebook) menggunakan CPU secara default di Windows, karena TensorFlow versi terbaru (≥2.11) tidak lagi mendukung native GPU/CUDA di Windows. Untuk mempercepat training dengan GPU NVIDIA, disarankan menggunakan WSL2 atau TensorFlow versi 2.10.

---

## 👤 Pengembang

Proyek tugas kuliah — dikembangkan sebagai eksperimen perbandingan pendekatan Deep Learning vs Machine Learning klasik untuk klasifikasi citra tekstur.
