# 🧅 Sortasi Mutu Bawang Merah Berbasis Computer Vision (CNN vs GLCM-SVM)

**Skripsi S1 Informatika** — Perbandingan performa **Convolutional Neural Network (CNN)** dan **GLCM + Support Vector Machine (SVM)** untuk klasifikasi mutu bawang merah (*Allium ascalonicum* L.) mengacu pada **SNI 3159:2013**, diimplementasikan ke dalam aplikasi desktop siap pakai.

> 📄 Penelitian ini telah disusun menjadi naskah jurnal ilmiah untuk publikasi pada jurnal terakreditasi SINTA: **JAITI** dan **JuTISI**, sebagai co-author bersama dosen pembimbing (Giatika Chrisnawati).

---

## 🎯 Latar Belakang

Sortasi mutu bawang merah pascapanen di Indonesia masih dominan dikerjakan manual — bersifat subjektif, tidak konsisten, dan rentan kesalahan. Standar Nasional Indonesia (**SNI 3159:2013**) membedakan Mutu I dan Mutu II berdasarkan cacat berskala milimeter yang sulit dideteksi mata telanjang secara konsisten, apalagi dalam skala industri.

Penelitian ini mengusulkan **otomasi berbasis Computer Vision**, menyederhanakan target klasifikasi menjadi dua kelas:
- **Layak** (gabungan Mutu I & Mutu II)
- **Tidak Layak**

---

## 🧪 Metodologi: Dua Pipeline yang Dibandingkan

| Pendekatan | Cara Kerja |
|---|---|
| **CNN** | Mempelajari fitur RGB spasial secara langsung dari citra berwarna (end-to-end deep learning) |
| **GLCM + SVM** | Ekstraksi fitur tekstur klasik (*contrast, correlation, energy, homogeneity*) dari citra grayscale, diklasifikasi dengan SVM kernel linear |

Kedua model dilatih dan diuji dengan skema yang identik (sumber data, jumlah data, proporsi split) agar perbandingan adil, lalu dievaluasi menggunakan **Accuracy, Precision, Recall, F1-Score, dan Confusion Matrix**.

---

## 📊 Hasil Penelitian

| Metrik | CNN | GLCM + SVM |
|---|---|---|
| **Akurasi** | **92,05%** | 54,30% |
| **F1-Score** | > 91% (kedua kelas) | Rendah |
| **Kesalahan klasifikasi** | 12 kasus | 69 kasus |
| **Kecepatan inferensi** | ~45 ms / citra (real-time) | — |

**Temuan utama:** CNN secara signifikan mengungguli GLCM-SVM. Konversi ke grayscale pada pipeline GLCM menghilangkan informasi warna (kromatik) yang justru krusial untuk mendeteksi tanda kebusukan pada bawang merah — sementara CNN mampu mempelajari fitur warna tersebut secara langsung. Hipotesis penelitian (H1: ada perbedaan akurasi signifikan antar metode) **terbukti diterima**.

---

## 📊 Dataset

- Sumber: data sekunder dari **Roboflow** dan **Kaggle**
- Total **3.590 citra** (hasil augmentasi), dibagi ke subset train/validation/test dengan proporsi seimbang antar kelas
- Pengujian akhir dilakukan pada **151 citra** test set
- Pra-pemrosesan: auto-orientation, resize 224×224
- Augmentasi: rotasi 90°, rotasi acak ±28°, salt-and-pepper noise

---

## 🖥️ Aplikasi Desktop (Implementasi)

Model CNN terbaik ditanam ke aplikasi GUI desktop (Tkinter) dengan mode **Batch Scanner**:
- Upload/scan banyak gambar bawang merah sekaligus
- Klasifikasi real-time (~45 ms per citra) dengan visualisasi hasil dari kedua model berdampingan
- Modal detail analisis: citra asli, grayscale, dan breakdown fitur GLCM
- `launcher.py` — installer otomatis yang mengecek & memasang seluruh dependency (numpy, opencv-python, tensorflow, scikit-learn, scikit-image, dll.) sebelum aplikasi utama dijalankan

---

## 📁 Struktur Proyek

```
├── Bawang_Merahipynb.ipynb              # Notebook: preprocessing, training CNN & GLCM-SVM, evaluasi
├── app.py                                # Aplikasi GUI utama (Tkinter) — mode Batch Scanner
├── launcher.py                           # Installer & pengecekan dependensi otomatis
├── jalankan.bat                          # Shortcut menjalankan aplikasi di Windows
├── Model_CNN_Bawang_Biner.h5             # Model CNN terlatih (klasifikasi biner)
├── Model_SVM_GLCM_Biner.pkl              # Model SVM+GLCM terlatih (klasifikasi biner)
├── Machine Learning Visual Texture..v3i.folder/   # Dataset (train/valid/test)
└── FODLER GAMBAR/                        # Contoh screenshot hasil aplikasi
```

---

## 🛠️ Teknologi

- **Python** — TensorFlow/Keras (CNN), Scikit-learn (SVM), Scikit-image (GLCM)
- **OpenCV & Pillow** — pemrosesan citra
- **Tkinter** — antarmuka aplikasi desktop
- **Google Colab** — environment eksperimen & training model

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

> Catatan: training menggunakan CPU secara default di Windows, karena TensorFlow ≥2.11 tidak lagi mendukung native GPU/CUDA di Windows. Disarankan WSL2 atau TensorFlow 2.10 untuk training via GPU NVIDIA.

---

## 🔭 Arah Pengembangan Selanjutnya

Berdasarkan masukan sidang skripsi, tiga arah pengembangan yang disarankan:
1. **Pemindaian real-time berbasis webcam**, melengkapi mode Batch Scanner saat ini
2. Pengujian **kernel non-linear (RBF/polinomial)** pada SVM untuk mengisolasi pengaruh pemilihan kernel terhadap performa GLCM
3. **Deploy ke perangkat edge** (mis. Raspberry Pi 5 + Edge TPU) untuk mesin sortir portabel di gudang/pasar induk

---

## 👤 Peneliti

**Juan Euaggelion Seran** — Program Studi Informatika, Universitas Bina Sarana Informatika
Dosen Pembimbing: Giatika Chrisnawati
