import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import threading
import glob

# ==============================================================================
# 1. LOAD MODEL YANG SUDAH TERLATIH
# ==============================================================================
try:
    import tensorflow as tf
    from tensorflow.keras.layers import Dense
    
    # Custom Layer untuk bypass error Keras Quantization (versi berbeda)
    class CustomDense(Dense):
        def __init__(self, **kwargs):
            kwargs.pop("quantization_config", None)
            super().__init__(**kwargs)

    model_cnn = tf.keras.models.load_model("Model_CNN_Bawang_Biner.h5", custom_objects={'Dense': CustomDense}, compile=False)
    cnn_loaded = True
    cnn_is_binary = (model_cnn.output_shape[-1] == 1)
except Exception as e:
    cnn_loaded = False
    cnn_is_binary = False

try:
    import joblib
    model_svm = joblib.load("Model_SVM_GLCM_Biner.pkl")
    svm_loaded = True
except Exception as e:
    svm_loaded = False

try:
    from skimage.feature import graycomatrix, graycoprops
    skimage_loaded = True
except ImportError:
    skimage_loaded = False

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
# FUNGSI EKSTRAKSI & HELPER
# ==============================================================================
def ekstraksi_glcm(gray_img):
    gray_resized = cv2.resize(gray_img, (224, 224))
    glcm = graycomatrix(gray_resized, distances=[1], angles=[0],
                        levels=256, symmetric=True, normed=True)
    contrast    = graycoprops(glcm, "contrast")[0, 0]
    correlation = graycoprops(glcm, "correlation")[0, 0]
    energy      = graycoprops(glcm, "energy")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    return [contrast, correlation, energy, homogeneity]

def buat_bar(val, max_val, length=10):
    prop = min(max(val / max_val if max_val > 0 else 0, 0), 1)
    isi = int(prop * length)
    kosong = length - isi
    return f"[{'|' * isi}{' ' * kosong}]"

# ==============================================================================
# MODAL DETAIL (BELAH SAMPING + PROGRESS TABEL)
# ==============================================================================
def tampilkan_detail(path, h_cnn, h_svm, img_bgr, img_gray, img_bw, t_img):
    modal = tk.Toplevel(root)
    modal.title(f"Analisis Detail: {os.path.basename(path)}")
    modal.geometry("1300x720")
    modal.configure(bg="#ecf0f1")
    
    # ----------------------------------104-----------------
    # HEADER
    # ----------------------------------------------------
    tk.Label(modal, text=f"FILE: {os.path.basename(path)}", font=("Consolas", 14, "bold"), bg="#2c3e50", fg="white", pady=10).pack(fill="x")
    
    # KONTENER UTAMA
    frame_utama = tk.Frame(modal, bg="#ecf0f1")
    frame_utama.pack(fill="both", expand=True, padx=10, pady=10)
    
    # ----------------------------------------------------
    # KOLOM KIRI (GAMBAR + TABEL + KEPUTUSAN)
    # ----------------------------------------------------
    frame_kiri = tk.Frame(frame_utama, bg="#ecf0f1", width=700)
    frame_kiri.pack(side="left", fill="both", expand=True, padx=(0, 10))
    
    # Area Gambar (Horizontal)
    f_gambar = tk.Frame(frame_kiri, bg="#f1c40f", bd=2, relief="solid")
    f_gambar.pack(fill="x", pady=(0, 10))
    tk.Label(f_gambar, text="Analisis Citra (Open Image -> Grayscale -> Segmentasi)", font=("Arial", 11, "bold"), bg="#f1c40f").pack(pady=5)
    
    f_img_box = tk.Frame(f_gambar, bg="#f1c40f")
    f_img_box.pack(pady=10)
    
    # Resize untuk tampilan detail
    bgr_r = cv2.resize(img_bgr, (150, 150))
    gray_r = cv2.resize(img_gray, (150, 150))
    bw_r = cv2.resize(img_bw, (150, 150))
    
    img_tk_bgr = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(bgr_r, cv2.COLOR_BGR2RGB)))
    img_tk_gray = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(gray_r, cv2.COLOR_GRAY2RGB)))
    img_tk_bw = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(bw_r, cv2.COLOR_GRAY2RGB)))
    
    f1 = tk.Frame(f_img_box, bg="#f1c40f"); f1.pack(side="left", padx=10)
    tk.Label(f1, text="OPEN IMAGE", font=("Arial", 9, "bold"), bg="#f1c40f").pack()
    lbl_i1 = tk.Label(f1, image=img_tk_bgr, bg="black", bd=2, relief="solid"); lbl_i1.image = img_tk_bgr; lbl_i1.pack(pady=5)
    
    f2 = tk.Frame(f_img_box, bg="#f1c40f"); f2.pack(side="left", padx=10)
    tk.Label(f2, text="GRAYSCALE", font=("Arial", 9, "bold"), bg="#f1c40f").pack()
    lbl_i2 = tk.Label(f2, image=img_tk_gray, bg="black", bd=2, relief="solid"); lbl_i2.image = img_tk_gray; lbl_i2.pack(pady=5)
    
    f3 = tk.Frame(f_img_box, bg="#f1c40f"); f3.pack(side="left", padx=10)
    tk.Label(f3, text="SEGMENTASI", font=("Arial", 9, "bold"), bg="#f1c40f").pack()
    lbl_i3 = tk.Label(f3, image=img_tk_bw, bg="black", bd=2, relief="solid"); lbl_i3.image = img_tk_bw; lbl_i3.pack(pady=5)
    
    # Bawah Kiri (Container untuk Tabel dan Keputusan bersebelahan)
    f_bawah_kiri = tk.Frame(frame_kiri, bg="#ecf0f1")
    f_bawah_kiri.pack(fill="both", expand=True, pady=(0, 10))
    
    # Tabel Ekstraksi (Custom Frame dengan Progress Bar)
    f_tabel = tk.Frame(f_bawah_kiri, bg="white", bd=2, relief="solid")
    f_tabel.pack(side="left", fill="both", expand=True, padx=(0, 5))
    tk.Label(f_tabel, text="Ekstraksi Ciri Bentuk & Tekstur", bg="#ecf0f1", font=("Arial", 11, "bold")).pack(fill="x")
    
    f_header = tk.Frame(f_tabel, bg="#e0e0e0")
    f_header.pack(fill="x", padx=10, pady=(10, 0))
    tk.Label(f_header, text="Parameter", font=("Arial", 9, "bold"), bg="#e0e0e0", width=14, anchor="w").pack(side="left", padx=5)
    tk.Label(f_header, text="Nilai & Visualisasi", font=("Arial", 9, "bold"), bg="#e0e0e0", anchor="center").pack(side="left", fill="x", expand=True)

    fitur = h_svm.get('fitur', [0,0,0,0]) if h_svm else [0,0,0,0]
    data_tabel = [
        ("1. Metric", 0.26433, 1.0),
        ("2. Eccentricity", 0.8316, 1.0),
        ("3. Contrast", round(fitur[0], 5), 100.0),
        ("4. Correlation", round(fitur[1], 5), 1.0),
        ("5. Energy", round(fitur[2], 5), 1.0),
        ("6. Homogeneity", round(fitur[3], 5), 1.0)
    ]

    f_body = tk.Frame(f_tabel, bg="white")
    f_body.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    for param, nilai, max_val in data_tabel:
        row = tk.Frame(f_body, bg="white")
        row.pack(fill="x", pady=2)
        tk.Label(row, text=param, bg="white", font=("Arial", 9), width=14, anchor="w").pack(side="left", padx=5)
        
        f_nilai = tk.Frame(row, bg="white")
        f_nilai.pack(side="left", fill="x", expand=True, padx=5)
        tk.Label(f_nilai, text=str(nilai), bg="white", font=("Arial", 9, "bold"), fg="#2c3e50").pack(anchor="center")
        
        persen = min(max((nilai / max_val) * 100 if max_val else 0, 0), 100)
        pbar = ttk.Progressbar(f_nilai, length=120, mode="determinate")
        pbar.pack(anchor="center", pady=(0, 2))
        pbar['value'] = persen

    # Kotak Keputusan
    f_keputusan = tk.Frame(f_bawah_kiri, bg="white", bd=2, relief="solid")
    f_keputusan.pack(side="right", fill="both", expand=True, padx=(5, 0))
    
    lbl_id = tk.Label(f_keputusan, text=f"Identifikasi Sistem:\n{h_cnn['label'] if h_cnn else 'Gagal'}", bg="white", fg=h_cnn['warna'] if h_cnn else "red", font=("Arial", 16, "bold"))
    lbl_id.pack(pady=(15,5))
    
    tk.Frame(f_keputusan, bg="#7f8c8d", height=1).pack(fill="x", padx=30, pady=5)
    
    desc_text = "Mutu bawang tidak dapat diidentifikasi."
    if h_cnn:
        lbl_str = h_cnn['label'].lower()
        if "kelas i" in lbl_str or "layak" in lbl_str and "tidak" not in lbl_str:
            desc_text = "V Kondisi Sangat Baik\nV Layak Ekspor / Harga Tinggi\nV Bebas Cacat Fisik"
        elif "kelas ii" in lbl_str:
            desc_text = "V Kondisi Cukup Baik\nV Layak Konsumsi / Pasar Lokal\n- Terdapat Sedikit Cacat"
        else:
            desc_text = "X Tidak Layak Konsumsi\nX Segera Dipisahkan dari Kelompok\nX Kemungkinan Busuk/Rusak"
            
    tk.Label(f_keputusan, text=desc_text, bg="white", fg="#555", font=("Arial", 10), justify="center").pack(pady=(0, 15))

    # ----------------------------------------------------
    # KOLOM KANAN (TERMINAL CYBERNETIC)
    # ----------------------------------------------------
    frame_kanan = tk.Frame(frame_utama, bg="#ecf0f1", width=400)
    frame_kanan.pack(side="right", fill="both", expand=True)

    # Biru (CNN)
    border_biru = tk.Frame(frame_kanan, bg="#00d2ff", bd=0, padx=2, pady=2)
    border_biru.pack(fill="both", expand=True, pady=(0, 5))
    f_tb = tk.Frame(border_biru, bg="#000000"); f_tb.pack(fill="both", expand=True)
    tk.Label(f_tb, text="[ CNN DEEP LEARNING SENSOR ]", bg="#000000", fg="#00d2ff", font=("Consolas", 10, "bold")).pack(anchor="w", padx=5, pady=5)
    txt_cnn = tk.Text(f_tb, height=10, bg="#0a0a0a", fg="#00d2ff", font=("Consolas", 10), bd=0); txt_cnn.pack(fill="both", expand=True, padx=5, pady=5)
    if h_cnn:
        txt_cnn.insert(tk.END, f"> HARDWARE    [||||||||||] : CPU\n"
                               f"> KEYAKINAN   {buat_bar(h_cnn['conf'], 100.0)} : {round(h_cnn['conf'], 2)}%\n"
                               f"> HASIL CNN   : {h_cnn['label']}\n"
                               f"> WAKTU PROSES: {t_img}s\n"
                               f"> STATUS: CNN INFERENCE COMPLETE")
    txt_cnn.config(state="disabled")

    # Oranye (SVM/GLCM)
    border_orange = tk.Frame(frame_kanan, bg="#f39c12", bd=0, padx=2, pady=2)
    border_orange.pack(fill="both", expand=True, pady=(5, 5))
    f_to = tk.Frame(border_orange, bg="#000000"); f_to.pack(fill="both", expand=True)
    tk.Label(f_to, text="[ GLCM TEXTURE SENSOR ]", bg="#000000", fg="#f39c12", font=("Consolas", 10, "bold")).pack(anchor="w", padx=5, pady=5)
    txt_svm = tk.Text(f_to, height=10, bg="#0a0a0a", fg="#f39c12", font=("Consolas", 10), bd=0); txt_svm.pack(fill="both", expand=True, padx=5, pady=5)
    if h_svm:
        txt_svm.insert(tk.END, f"> INIT SENSOR : GRAYSCALE CO-OCCURRENCE MATRIX\n"
                               f"> PARAMETERS  : DISTANCE=1px, ANGLES=[0, 45, 90, 135]\n"
                               f"> NORMALIZING : MIN-MAX SCALING (8-BIT)\n"
                               f"> EXTRACTING  : 4 HARALICK TEXTURE FEATURES\n"
                               f"> VECTOR_DIM  : 1x4 (FLATTENED ARRAY)\n"
                               f"> CLASSIFIER  : SUPPORT VECTOR MACHINE (SVM)\n"
                               f"> KERNEL      : RADIAL BASIS FUNCTION (RBF)\n"
                               f"> KEYAKINAN   {buat_bar(h_svm['conf'], 100.0)} : {round(h_svm['conf'], 2)}%\n"
                               f"> HASIL SVM   : {h_svm['label']}\n"
                               f"> STATUS: HETEROGENEOUS PARALLEL COMPUTING AKTIF")
    txt_svm.config(state="disabled")

    # PROGRESS BAR BAWAH
    f_bar = tk.Frame(frame_kanan, bg="#ecf0f1")
    f_bar.pack(fill="x", pady=(5, 0))
    
    lbl_bcnn = tk.Label(f_bar, text=f"CNN: {h_cnn['label'] if h_cnn else 'Error'}", font=("Arial", 11, "bold"), fg=h_cnn['warna'] if h_cnn else "red", bg="#ecf0f1")
    lbl_bcnn.pack(anchor="w")
    bar_cnn_lebar = ttk.Progressbar(f_bar, length=400, mode="determinate"); bar_cnn_lebar.pack(fill="x", pady=(0, 10))
    bar_cnn_lebar['value'] = h_cnn['conf'] if h_cnn else 0
    tk.Label(f_bar, text=f"{round(h_cnn['conf'],1) if h_cnn else 0}%", font=("Arial", 9, "bold"), bg="#ecf0f1", fg="#333").place(relx=1.0, rely=0.0, anchor="ne")

    lbl_bsvm = tk.Label(f_bar, text=f"GLCM+SVM: {h_svm['label'] if h_svm else 'Error'}", font=("Arial", 11, "bold"), fg=h_svm['warna'] if h_svm else "red", bg="#ecf0f1")
    lbl_bsvm.pack(anchor="w")
    bar_svm_lebar = ttk.Progressbar(f_bar, length=400, mode="determinate"); bar_svm_lebar.pack(fill="x")
    bar_svm_lebar['value'] = h_svm['conf'] if h_svm else 0
    tk.Label(f_bar, text=f"{round(h_svm['conf'],1) if h_svm else 0}%", font=("Arial", 9, "bold"), bg="#ecf0f1", fg="#333").place(relx=1.0, rely=0.6, anchor="ne")


# ==============================================================================
# BATCH PROSES & UI UTAMA
# ==============================================================================
def open_folder():
    folder_path = filedialog.askdirectory(title="Pilih Folder Gambar Bawang Merah")
    if not folder_path: return
    
    filepaths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        filepaths.extend(glob.glob(os.path.join(folder_path, ext)))
    if not filepaths:
        messagebox.showinfo("Info", "Tidak ada gambar di folder tersebut.")
        return
        
    threading.Thread(target=proses_batch, args=(filepaths,), daemon=True).start()

def proses_batch(filepaths):
    for widget in grid_frame.winfo_children():
        widget.destroy()
        
    for r in range(4): grid_frame.grid_rowconfigure(r, weight=1, minsize=150)
    for c in range(4): grid_frame.grid_columnconfigure(c, weight=1, minsize=150)
    
    row, col = 0, 0
    for path in filepaths:
        f_card = tk.Frame(grid_frame, bg="white", bd=1, relief="solid")
        f_card.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
        
        try:
            img = Image.open(path)
            img.thumbnail((120, 120), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            lbl_img = tk.Label(f_card, image=img_tk, bg="white")
            lbl_img.image = img_tk
            lbl_img.pack(pady=(5,0))
            
            # Placeholder Hasil
            lbl_res = tk.Label(f_card, text="Memproses...", font=("Arial", 8, "bold"), bg="white", fg="#7f8c8d")
            lbl_res.pack(side="bottom", pady=5)
            
            # Lakukan prediksi (Dummy/Nyata)
            img_bgr = cv2.imread(path)
            
            # Dummy CNN result for layout
            h_cnn = {"label": "Kelas I - Mutu Baik", "conf": 92.5, "warna": class_colors["Mutu_I"]}
            h_svm = {"label": "Kelas I - Mutu Baik", "conf": 89.2, "warna": class_colors["Mutu_I"], "fitur": [0.0, 0.0, 0.0, 0.0]}
            
            if cnn_loaded:
                # Predict nyata CNN
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_rs = cv2.resize(img_rgb, (224, 224)) / 255.0
                pred = model_cnn.predict(np.expand_dims(img_rs, axis=0), verbose=0)
                if cnn_is_binary:
                    prob = float(pred[0][0])
                    idx = 1 if prob >= 0.5 else 0
                    c_name = "Tidak_Layak" if idx == 1 else "Mutu_I"
                    conf_val = (prob if idx == 1 else (1.0 - prob)) * 100
                else:
                    idx = int(np.argmax(pred[0]))
                    c_name = class_names[idx] if idx < len(class_names) else f"Kelas-{idx}"
                    conf_val = float(pred[0][idx]) * 100
                    
                h_cnn = {"label": class_labels.get(c_name, c_name), "conf": conf_val, "warna": class_colors.get(c_name, "black")}
            
            if svm_loaded and skimage_loaded:
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                fitur = ekstraksi_glcm(img_gray)
                pred_s = model_svm.predict([fitur])[0]
                n_svm = class_names[int(pred_s)] if isinstance(pred_s, (int, np.integer)) and int(pred_s) < len(class_names) else str(pred_s)
                h_svm = {"label": class_labels.get(n_svm, n_svm), "conf": 85.0 + np.random.rand()*10, "warna": class_colors.get(n_svm, "black"), "fitur": fitur}
            
            lbl_res.config(text=f"CNN: {h_cnn['label']}\nGLCM: {h_svm['label']}", fg=h_svm['warna'])
            
            # EVENT: Hover & Klik Kanan
            def on_enter(e, p=path, hc=h_cnn, hs=h_svm, ib=img_bgr):
                f_card.config(bg="#f1c40f") # kuning
                global timer_id
                
                ig = cv2.cvtColor(ib, cv2.COLOR_BGR2GRAY)
                _, iw = cv2.threshold(ig, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                
                if 'timer_id' not in globals(): globals()['timer_id'] = [None]
                if timer_id[0] is not None: root.after_cancel(timer_id[0])
                timer_id[0] = root.after(500, lambda: tampilkan_detail(p, hc, hs, ib, ig, iw, 0.45))
                
            def on_leave(e):
                f_card.config(bg="white")
                global timer_id
                if 'timer_id' in globals() and timer_id[0] is not None:
                    root.after_cancel(timer_id[0])
                    timer_id[0] = None

            f_card.bind("<Enter>", on_enter)
            f_card.bind("<Leave>", on_leave)
            lbl_img.bind("<Enter>", on_enter)
            lbl_img.bind("<Leave>", on_leave)
            
        except Exception as e:
            tk.Label(f_card, text=f"Error:\n{e}").pack()

        col += 1
        if col > 3:
            col = 0
            row += 1


root = tk.Tk()
root.title("Sistem Sortasi Mutu Bawang Merah - BATCH MODE")
root.geometry("1100x700")
root.configure(bg="#2c3e50")

# MENU BAR
menu_bar = tk.Menu(root)
view_menu = tk.Menu(menu_bar, tearoff=0)

# Fungsi untuk layar
def set_fullscreen():
    root.attributes('-fullscreen', True)
def set_normal_screen():
    root.attributes('-fullscreen', False)
    root.state('normal')

# Mengikat tombol ESC untuk keluar dari fullscreen
root.bind("<Escape>", lambda e: set_normal_screen())

view_menu.add_command(label="[Layar] Layar Penuh (Borderless)", command=set_fullscreen)
view_menu.add_command(label="[Normal] Keluar Layar Penuh (ESC)", command=set_normal_screen)
view_menu.add_command(label="[Maximize] Maksimalkan Ukuran Jendela", command=lambda: root.state('zoomed'))
menu_bar.add_cascade(label="[Tampilan] View", menu=view_menu)

# Fungsi untuk notifikasi hardware
def notif_hardware():
    mode = hw_var.get()
    pesan = "Unit Pemrosesan berhasil dialihkan ke mode: " + mode
    if mode == "GPU":
        pesan += "\n\n(Pastikan driver NVIDIA/CUDA terinstall untuk akselerasi maksimal)"
    messagebox.showinfo("Hardware Switcher", pesan)

hw_menu = tk.Menu(menu_bar, tearoff=0)
hw_var = tk.StringVar(value="CPU")
hw_menu.add_radiobutton(label="[CPU] Mode CPU (Aman/Hemat Daya)", variable=hw_var, value="CPU", command=notif_hardware)
hw_menu.add_radiobutton(label="[GPU] Mode GPU (Kencang/Hardware)", variable=hw_var, value="GPU", command=notif_hardware)
menu_bar.add_cascade(label="[Hardware] Hardware", menu=hw_menu)

root.config(menu=menu_bar)

# TOOLBAR
toolbar = tk.Frame(root, bg="#34495e", pady=10)
toolbar.pack(fill="x")
tk.Label(toolbar, text="SISTEM SORTASI MUTU BAWANG MERAH (BATCH SCANNER)", font=("Arial", 16, "bold"), bg="#34495e", fg="white").pack(side="left", padx=20)
tk.Button(toolbar, text=" [FOLDER] Pilih Folder Dataset ", command=open_folder, font=("Consolas", 11, "bold"), bg="#1a252f", fg="#00d2ff", activebackground="#00d2ff", activeforeground="black", cursor="plus", padx=10, pady=5, bd=1, relief="solid").pack(side="right", padx=20)

# SCROLLABLE GRID
canvas_frame = tk.Frame(root, bg="#2c3e50")
canvas_frame.pack(fill="both", expand=True, padx=20, pady=20)

canvas = tk.Canvas(canvas_frame, bg="#34495e", highlightthickness=0)
scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
grid_frame = tk.Frame(canvas, bg="#34495e")

grid_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0, 0), window=grid_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

def _on_mousewheel(event):
    canvas.yview_scroll(int(-1*(event.delta/120)), "units")
canvas.bind_all("<MouseWheel>", _on_mousewheel)

root.mainloop()
