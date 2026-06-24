import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import threading

# ==============================================================================
# 1. LOAD MODEL YANG SUDAH TERLATIH
# ==============================================================================
try:
    import tensorflow as tf
    model_cnn = tf.keras.models.load_model("Model_CNN_Bawang_Final.h5")
    cnn_loaded = True
    cnn_error = ""
except Exception as e:
    cnn_loaded = False
    cnn_error = str(e)

try:
    import joblib
    model_svm = joblib.load("Model_SVM_GLCM.pkl")
    svm_loaded = True
except Exception as e:
    svm_loaded = False
    svm_error = str(e)

try:
    from skimage.feature import graycomatrix, graycoprops
    skimage_loaded = True
except ImportError:
    skimage_loaded = False

# ==============================================================================
# NAMA KELAS MUTU BAWANG (sesuai SNI 3159:2013)
# Sesuaikan urutan ini dengan urutan folder saat training model CNN!
# ==============================================================================
class_names = ["Mutu_I", "Mutu_II", "Tidak_Layak"]
class_labels = {
    "Mutu_I":      "Kelas I - Mutu Baik",
    "Mutu_II":     "Kelas II - Mutu Sedang",
    "Tidak_Layak": "Tidak Layak / Mutu Rendah",
}
class_colors = {
    "Mutu_I":      "#2ecc71",
    "Mutu_II":     "#f39c12",
    "Tidak_Layak": "#e74c3c",
}

# ==============================================================================
# 2. FUNGSI PEMROSESAN DAN PREDIKSI
# ==============================================================================

def ekstraksi_glcm(gray_img):
    """Ekstraksi 4 fitur tekstur GLCM dari gambar grayscale."""
    gray_resized = cv2.resize(gray_img, (224, 224))
    glcm = graycomatrix(gray_resized, distances=[1], angles=[0],
                        levels=256, symmetric=True, normed=True)
    contrast    = graycoprops(glcm, "contrast")[0, 0]
    correlation = graycoprops(glcm, "correlation")[0, 0]
    energy      = graycoprops(glcm, "energy")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    return [contrast, correlation, energy, homogeneity]


def jalankan_prediksi(filepath):
    try:
        img_bgr = cv2.imread(filepath)
        if img_bgr is None:
            messagebox.showerror("Error", "Gagal membaca gambar:\n" + filepath)
            return

        # --- PROSES MODEL CNN ---
        if cnn_loaded:
            img_rgb        = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resize_cnn = cv2.resize(img_rgb, (224, 224))
            img_norm_cnn   = img_resize_cnn / 255.0
            img_input_cnn  = np.expand_dims(img_norm_cnn, axis=0)
            pred_cnn_prob  = model_cnn.predict(img_input_cnn, verbose=0)
            idx_cnn        = int(np.argmax(pred_cnn_prob[0]))
            nama_kelas_cnn = class_names[idx_cnn] if idx_cnn < len(class_names) else ("Kelas-" + str(idx_cnn))
            label_cnn      = class_labels.get(nama_kelas_cnn, nama_kelas_cnn)
            conf_cnn       = float(pred_cnn_prob[0][idx_cnn]) * 100
            warna_cnn      = class_colors.get(nama_kelas_cnn, "#2980b9")
            label_hasil_cnn.config(
                text="Hasil CNN:  " + label_cnn + "  (" + str(round(conf_cnn, 2)) + "%)",
                fg=warna_cnn
            )
            bar_cnn["value"] = conf_cnn
            lbl_bar_cnn.config(text=str(round(conf_cnn, 1)) + "%")
        else:
            label_hasil_cnn.config(text="Model CNN tidak ditemukan", fg="#e74c3c")

        # --- PROSES MODEL GLCM + SVM ---
        if svm_loaded and skimage_loaded:
            img_gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            fitur_glcm = ekstraksi_glcm(img_gray)
            pred_svm   = model_svm.predict([fitur_glcm])[0]
            if isinstance(pred_svm, (int, np.integer)) and int(pred_svm) < len(class_names):
                nama_svm = class_names[int(pred_svm)]
            else:
                nama_svm = str(pred_svm)
            label_svm = class_labels.get(nama_svm, nama_svm)
            warna_svm = class_colors.get(nama_svm, "#8e44ad")
            label_hasil_glcm.config(text="Hasil GLCM+SVM:  " + label_svm, fg=warna_svm)
            fitur_text = ("Fitur Tekstur GLCM:\n"
                         + "  Contrast    : " + str(round(fitur_glcm[0], 6)) + "\n"
                         + "  Correlation : " + str(round(fitur_glcm[1], 6)) + "\n"
                         + "  Energy      : " + str(round(fitur_glcm[2], 6)) + "\n"
                         + "  Homogeneity : " + str(round(fitur_glcm[3], 6)))
            label_info_glcm.config(text=fitur_text)
        else:
            label_hasil_glcm.config(text="Model SVM tidak ditemukan", fg="#e74c3c")
            label_info_glcm.config(text="Fitur GLCM: tidak dapat dihitung")

        label_status.config(text="Prediksi selesai.", fg="#27ae60")

    except Exception as e:
        messagebox.showerror("Error Prediksi", "Terjadi kesalahan:\n" + str(e))
        label_status.config(text="Error: " + str(e), fg="#e74c3c")


# ==============================================================================
# 3. MEMBANGUN ANTARMUKA APLIKASI (GUI)
# ==============================================================================

def open_image():
    file_path = filedialog.askopenfilename(
        title="Pilih Gambar Bawang Merah",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not file_path:
        return
    label_status.config(text="Memproses gambar...", fg="#f39c12")
    label_hasil_cnn.config(text="Hasil CNN: memproses...", fg="#555555")
    label_hasil_glcm.config(text="Hasil GLCM+SVM: memproses...", fg="#555555")
    label_info_glcm.config(text="Fitur GLCM: memproses...")
    bar_cnn["value"] = 0
    lbl_bar_cnn.config(text="0%")
    root.update()
    label_nama_file.config(text="File: " + os.path.basename(file_path))
    try:
        img = Image.open(file_path)
        img_display = img.copy()
        img_display.thumbnail((300, 300), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_display)
        canvas_image.config(image=img_tk, text="", bg="#dfe6e9")
        canvas_image.image = img_tk
    except Exception as e:
        messagebox.showerror("Error", "Gagal membuka gambar:\n" + str(e))
        return
    t = threading.Thread(
        target=lambda: [jalankan_prediksi(file_path), root.update()],
        daemon=True
    )
    t.start()


# --- Buat Window Utama ---
root = tk.Tk()
root.title("Sistem Sortasi Mutu Bawang Merah - SNI 3159:2013")
root.geometry("640x720")
root.resizable(False, False)
root.configure(bg="#f0f3f4")

# Header
frame_header = tk.Frame(root, bg="#2c3e50", pady=12)
frame_header.pack(fill="x")
tk.Label(frame_header, text="SORTASI MUTU BAWANG MERAH",
         font=("Arial", 16, "bold"), bg="#2c3e50", fg="white").pack()
tk.Label(frame_header, text="Metode GLCM+SVM dan CNN  |  SNI 3159:2013",
         font=("Arial", 10), bg="#2c3e50", fg="#bdc3c7").pack()

# Area Gambar
frame_gambar = tk.Frame(root, bg="#f0f3f4", pady=8)
frame_gambar.pack()
canvas_image = tk.Label(frame_gambar, text="[ Upload gambar bawang merah di sini ]",
                        bg="#dfe6e9", fg="#7f8c8d", font=("Arial", 10),
                        width=40, height=15, relief="groove", bd=2)
canvas_image.pack()
label_nama_file = tk.Label(frame_gambar, text="", font=("Arial", 8),
                           bg="#f0f3f4", fg="#7f8c8d")
label_nama_file.pack()

# Tombol Upload
btn_upload = tk.Button(root, text="  Pilih Gambar Bawang  ", command=open_image,
                       font=("Arial", 11, "bold"), bg="#27ae60", fg="white",
                       activebackground="#2ecc71", activeforeground="white",
                       padx=16, pady=7, relief="flat", cursor="hand2")
btn_upload.pack(pady=(0, 10))

# Frame Hasil Prediksi
frame_hasil = tk.Frame(root, bg="white", bd=1, relief="groove", padx=15, pady=12)
frame_hasil.pack(fill="x", padx=20, pady=5)
tk.Label(frame_hasil, text="HASIL PREDIKSI", font=("Arial", 10, "bold"),
         bg="white", fg="#2c3e50").pack(anchor="w")
ttk.Separator(frame_hasil, orient="horizontal").pack(fill="x", pady=5)

label_hasil_cnn = tk.Label(frame_hasil, text="CNN: -", font=("Arial", 12, "bold"),
                           bg="white", fg="#7f8c8d", anchor="w")
label_hasil_cnn.pack(fill="x")

frame_bar = tk.Frame(frame_hasil, bg="white")
frame_bar.pack(fill="x", pady=(2, 6))
tk.Label(frame_bar, text="   Confidence:", font=("Arial", 9), bg="white", fg="#555").pack(side="left")
bar_cnn = ttk.Progressbar(frame_bar, length=380, mode="determinate", maximum=100)
bar_cnn.pack(side="left", padx=6)
lbl_bar_cnn = tk.Label(frame_bar, text="0%", font=("Arial", 9, "bold"),
                       bg="white", fg="#2c3e50", width=5)
lbl_bar_cnn.pack(side="left")

label_hasil_glcm = tk.Label(frame_hasil, text="GLCM+SVM: -", font=("Arial", 12, "bold"),
                            bg="white", fg="#7f8c8d", anchor="w")
label_hasil_glcm.pack(fill="x")

# Frame Fitur GLCM
frame_glcm = tk.Frame(root, bg="#eaf4fb", bd=1, relief="groove", padx=15, pady=10)
frame_glcm.pack(fill="x", padx=20, pady=5)
label_info_glcm = tk.Label(frame_glcm, text="Fitur Tekstur GLCM: -",
                           font=("Courier", 9), bg="#eaf4fb", fg="#2c3e50",
                           justify="left", anchor="w")
label_info_glcm.pack(fill="x")

# Footer / Status Bar
frame_footer = tk.Frame(root, bg="#ecf0f1", pady=5)
frame_footer.pack(fill="x", side="bottom")
label_status = tk.Label(frame_footer, text="Siap. Pilih gambar untuk memulai prediksi.",
                        font=("Arial", 9), bg="#ecf0f1", fg="#7f8c8d")
label_status.pack()
tk.Label(frame_footer, text="Skripsi - BSI Dewi Sartika  |  TA 2025-2026",
         font=("Arial", 8), bg="#ecf0f1", fg="#bdc3c7").pack()

# Cek status model saat startup
status_parts = []
if not cnn_loaded:
    status_parts.append("Model CNN tidak ditemukan (Model_CNN_Bawang_Final.h5)")
if not svm_loaded:
    status_parts.append("Model SVM tidak ditemukan (Model_SVM_GLCM.pkl)")
if not skimage_loaded:
    status_parts.append("scikit-image tidak terinstall  ->  pip install scikit-image")
if status_parts:
    messagebox.showwarning("Peringatan - Model Tidak Ditemukan",
                           "\n".join(status_parts) +
                           "\n\nPastikan file model berada di folder yang sama dengan app.py")
else:
    label_status.config(text="Semua model berhasil dimuat. Siap untuk prediksi.", fg="#27ae60")

root.mainloop()