#!/usr/bin/env python3
# =============================================================================
# LAUNCHER - SISTEM SORTASI MUTU BAWANG MERAH AI
# Analisis Dependensi & Auto-Installer
# Jalankan file ini PERTAMA KALI sebelum menggunakan aplikasi utama
# =============================================================================
import os, sys, subprocess, importlib, threading, time, tkinter as tk
from tkinter import ttk

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
LOG_FILE  = os.path.join(BASE_DIR, "app_error.log")
APP_FILE  = os.path.join(BASE_DIR, "app.py")
PYTHON    = sys.executable

# ==============================================================================
# DAFTAR DEPENDENSI YANG DIPERLUKAN (modul_import, pip_name, keterangan)
# ==============================================================================
PACKAGES = [
    ("numpy",    "numpy",          "Komputasi Numerik Matriks"),
    ("cv2",      "opencv-python",  "Computer Vision / OpenCV"),
    ("PIL",      "Pillow",         "Pemrosesan Gambar (PIL/Pillow)"),
    ("sklearn",  "scikit-learn",   "Machine Learning (SVM Classifier)"),
    ("skimage",  "scikit-image",   "Ekstraksi Fitur Tekstur GLCM"),
    ("joblib",   "joblib",         "Serialisasi Model (.pkl)"),
    ("tensorflow", None,           "TensorFlow Deep Learning (CNN)"),  # pip_name=None = hanya cek
]

MODEL_FILES = [
    ("Model_CNN_Fixed.h5",    "Model CNN (utama)"),
    ("Model_CNN_Bawang_Final.h5", "Model CNN (alternatif)"),
    ("Model_SVM_GLCM.pkl",    "Model SVM+GLCM"),
]

# ==============================================================================
# FUNGSI LOG
# ==============================================================================
def tulis_log(tipe, pesan):
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    label = {"INFO": "INFO    ", "WARNING": "WARNING ", "ERROR": "ERROR   ", "OK": "OK      "}.get(tipe, tipe)
    baris = f"{ts}  [{label}]  {pesan}\n"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(baris)
    except Exception:
        pass
    return baris.rstrip("\n")

# ==============================================================================
# JENDELA LAUNCHER
# ==============================================================================
class LauncherWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🔧 Diagnostic & Setup — Bawang Merah AI")
        self.root.geometry("820x560")
        self.root.configure(bg="#0d1117")
        self.root.resizable(False, False)

        self.selesai = False
        self.sukses  = False
        self._blink_state = True
        self._auto_refresh = True

        self._build_ui()
        self._mulai_diagnostik()

    def _build_ui(self):
        # ── Header ──────────────────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg="#161b22", pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr, text="⬡  SISTEM SORTASI MUTU BAWANG MERAH AI",
                 bg="#161b22", fg="#00d2ff",
                 font=("Consolas", 14, "bold")).pack(side="left", padx=15)
        self.lbl_live = tk.Label(hdr, text="◉ LIVE SCANNING",
                                 bg="#161b22", fg="#2ecc71",
                                 font=("Consolas", 10, "bold"))
        self.lbl_live.pack(side="right", padx=15)

        # ── Label status ────────────────────────────────────────────────────
        self.lbl_status = tk.Label(self.root,
            text="> MEMULAI ANALISIS SISTEM...",
            bg="#0d1117", fg="#f1c40f",
            font=("Consolas", 11, "bold"), anchor="w")
        self.lbl_status.pack(fill="x", padx=15, pady=(10, 2))

        # ── Progress bar ────────────────────────────────────────────────────
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Cyber.Horizontal.TProgressbar",
                         troughcolor="#1c2128", background="#00d2ff",
                         bordercolor="#0d1117", lightcolor="#00d2ff",
                         darkcolor="#0080b0")
        self.progress = ttk.Progressbar(self.root,
            style="Cyber.Horizontal.TProgressbar",
            length=780, mode="determinate", maximum=100)
        self.progress.pack(padx=15, pady=4)

        # ── Legenda warna ───────────────────────────────────────────────────
        leg = tk.Frame(self.root, bg="#0d1117")
        leg.pack(fill="x", padx=15)
        for warna, label in [("#2ecc71","● OK"), ("#e74c3c","● ERROR"),
                              ("#f39c12","● WARNING"), ("#00d2ff","● INFO"),
                              ("#aaaaaa","● NORMAL")]:
            tk.Label(leg, text=label, bg="#0d1117", fg=warna,
                     font=("Consolas", 8, "bold")).pack(side="left", padx=6)

        # ── Area log terminal ───────────────────────────────────────────────
        frm = tk.Frame(self.root, bg="#0d1117")
        frm.pack(fill="both", expand=True, padx=15, pady=(4, 5))
        sb = tk.Scrollbar(frm); sb.pack(side="right", fill="y")
        self.txt = tk.Text(frm,
            bg="#0d1117", fg="#c9d1d9",
            font=("Consolas", 9), bd=0, state="disabled",
            wrap="none", yscrollcommand=sb.set)
        self.txt.pack(fill="both", expand=True)
        sb.config(command=self.txt.yview)
        self.txt.tag_configure("ok",      foreground="#2ecc71", font=("Consolas", 9, "bold"))
        self.txt.tag_configure("error",   foreground="#e74c3c", font=("Consolas", 9, "bold"))
        self.txt.tag_configure("warning", foreground="#f39c12", font=("Consolas", 9))
        self.txt.tag_configure("info",    foreground="#00d2ff", font=("Consolas", 9))
        self.txt.tag_configure("sep",     foreground="#30363d", font=("Consolas", 9))
        self.txt.tag_configure("normal",  foreground="#c9d1d9", font=("Consolas", 9))

        # ── Tombol bawah ────────────────────────────────────────────────────
        btn_frm = tk.Frame(self.root, bg="#161b22", pady=8)
        btn_frm.pack(fill="x")
        self.btn_launch = tk.Button(btn_frm,
            text="⏳  Harap Tunggu...",
            bg="#30363d", fg="#888888",
            font=("Consolas", 11, "bold"),
            padx=25, pady=6, relief="flat", cursor="arrow",
            state="disabled")
        self.btn_launch.pack(side="right", padx=15)
        tk.Button(btn_frm, text="✖ Keluar",
            bg="#c0392b", fg="white",
            font=("Consolas", 10, "bold"),
            padx=15, pady=6, relief="flat", cursor="hand2",
            command=self.root.destroy).pack(side="right", padx=5)
        self.lbl_hitungan = tk.Label(btn_frm,
            text="Dependensi OK: 0 | Error: 0 | Warning: 0",
            bg="#161b22", fg="#aaaaaa",
            font=("Consolas", 9))
        self.lbl_hitungan.pack(side="left", padx=15)

    # ── Tambah baris ke terminal ──────────────────────────────────────────
    def tambah_baris(self, teks, tag="normal"):
        self.txt.config(state="normal")
        self.txt.insert(tk.END, teks + "\n", tag)
        self.txt.see(tk.END)
        self.txt.config(state="disabled")

    # ── Update status label ───────────────────────────────────────────────
    def set_status(self, teks, warna="#f1c40f"):
        self.lbl_status.config(text=f"> {teks}", fg=warna)

    # ── Blinking LIVE indicator ───────────────────────────────────────────
    def _blink(self):
        if not self._auto_refresh: return
        self._blink_state = not self._blink_state
        self.lbl_live.config(
            fg="#2ecc71" if self._blink_state else "#0d1117",
            text="◉ LIVE SCANNING")
        self.root.after(500, self._blink)

    # ── Mulai diagnostik di thread terpisah ──────────────────────────────
    def _mulai_diagnostik(self):
        self._blink()
        threading.Thread(target=self._jalankan_diagnostik, daemon=True).start()

    def _jalankan_diagnostik(self):
        ok_count = 0; err_count = 0; warn_count = 0

        def log(tipe, pesan, tag="normal"):
            nonlocal ok_count, err_count, warn_count
            baris = tulis_log(tipe, pesan)
            self.root.after(0, self.tambah_baris, baris, tag)
            if tipe == "OK":      ok_count  += 1
            elif tipe == "ERROR": err_count += 1
            elif tipe == "WARNING": warn_count += 1
            self.root.after(0, self.lbl_hitungan.config,
                {"text": f"Dependensi OK: {ok_count} | Error: {err_count} | Warning: {warn_count}"})
            time.sleep(0.05)

        # Header log
        tulis_log("INFO", "=" * 55)
        tulis_log("INFO", "LAUNCHER DIAGNOSTIC — Sistem Bawang Merah AI")
        tulis_log("INFO", f"Python : {sys.version.split()[0]}  |  Path: {PYTHON}")
        tulis_log("INFO", "=" * 55)
        self.root.after(0, self.tambah_baris,
            "=" * 70 + "\n  LAUNCHER DIAGNOSTIC — SISTEM SORTASI MUTU BAWANG MERAH AI\n" + "=" * 70, "info")

        total = len(PACKAGES) + len(MODEL_FILES) + 1
        langkah = 0

        # ── TAHAP 1: Cek versi Python ─────────────────────────────────────
        self.root.after(0, self.set_status, "Menganalisis versi Python...")
        v = sys.version_info
        if v.major == 3 and v.minor >= 8:
            log("OK", f"Python {v.major}.{v.minor}.{v.micro} ✔  (Kompatibel)", "ok")
        else:
            log("WARNING", f"Python {v.major}.{v.minor} — Disarankan Python 3.8+", "warning")
        langkah += 1
        self.root.after(0, self.progress.config, {"value": int(langkah/total*100)})

        # ── TAHAP 2: Cek & Install Dependensi ────────────────────────────
        self.root.after(0, self.tambah_baris, "\n  [ ANALISIS DEPENDENSI ]", "info")
        to_install = []

        for modul, pip_name, ket in PACKAGES:
            self.root.after(0, self.set_status, f"Menganalisis: {ket}...")
            try:
                importlib.import_module(modul)
                log("OK", f"{modul:<12} ✔  {ket}", "ok")
            except ImportError:
                if pip_name is None:
                    log("WARNING", f"{modul:<12} ✘  {ket} — TIDAK TERDETEKSI (Install manual!)", "warning")
                    warn_count += 1
                else:
                    log("ERROR", f"{modul:<12} ✘  {ket} — AKAN DIINSTALL", "error")
                    to_install.append((modul, pip_name, ket))
            langkah += 1
            self.root.after(0, self.progress.config, {"value": int(langkah/total*100)})

        # ── TAHAP 3: Cek File Model ───────────────────────────────────────
        self.root.after(0, self.tambah_baris, "\n  [ ANALISIS FILE MODEL ]", "info")
        model_ok = False
        for fname, ket in MODEL_FILES:
            fpath = os.path.join(BASE_DIR, fname)
            self.root.after(0, self.set_status, f"Mencari: {fname}...")
            if os.path.exists(fpath):
                ukuran = os.path.getsize(fpath) // (1024*1024)
                log("OK", f"{fname:<35} ✔  {ket}  [{ukuran} MB]", "ok")
                model_ok = True
            else:
                log("WARNING", f"{fname:<35} ✘  {ket} — FILE TIDAK DITEMUKAN", "warning")
            langkah += 1
            self.root.after(0, self.progress.config, {"value": int(langkah/total*100)})

        if not model_ok:
            log("ERROR", "PERINGATAN: Tidak ada file model .h5 atau .pkl ditemukan!", "error")
            log("WARNING", "Letakkan Model_CNN_Fixed.h5 dan Model_SVM_GLCM.pkl di folder ini.", "warning")

        # ── TAHAP 4: Auto-Install ─────────────────────────────────────────
        if to_install:
            self.root.after(0, self.tambah_baris, "\n  [ AUTO-INSTALL DEPENDENSI ]", "info")
            for modul, pip_name, ket in to_install:
                self.root.after(0, self.set_status, f"Menginstall: {pip_name}...", "#f39c12")
                log("INFO", f"pip install {pip_name} ...", "info")
                try:
                    result = subprocess.run(
                        [PYTHON, "-m", "pip", "install", pip_name, "--quiet"],
                        capture_output=True, text=True, timeout=120)
                    if result.returncode == 0:
                        log("OK", f"{pip_name} berhasil diinstall ✔", "ok")
                    else:
                        log("ERROR", f"Gagal install {pip_name}: {result.stderr[:80]}", "error")
                except subprocess.TimeoutExpired:
                    log("ERROR", f"Timeout saat install {pip_name}", "error")
                except Exception as e:
                    log("ERROR", f"Exception install {pip_name}: {e}", "error")
        else:
            self.root.after(0, self.tambah_baris,
                "\n  [ SEMUA DEPENDENSI TERSEDIA — TIDAK ADA YANG PERLU DIINSTALL ]", "ok")

        # ── TAHAP 5: Selesai ──────────────────────────────────────────────
        self.root.after(0, self.progress.config, {"value": 100})
        tulis_log("INFO", "=" * 55)
        tulis_log("INFO", f"DIAGNOSTIK SELESAI | OK: {ok_count} | Error: {err_count} | Warning: {warn_count}")
        tulis_log("INFO", "=" * 55)
        self.root.after(0, self.tambah_baris,
            f"\n{'=' * 70}\n  DIAGNOSTIK SELESAI — OK: {ok_count}  ERROR: {err_count}  WARNING: {warn_count}\n{'=' * 70}",
            "ok" if err_count == 0 else "warning")

        self.selesai = True
        self.sukses  = (err_count == 0)
        self._auto_refresh = False  # MATIKAN auto-refresh

        warna_status = "#2ecc71" if self.sukses else "#f39c12"
        teks_status  = "DIAGNOSTIK SELESAI — Siap Meluncurkan Aplikasi" if self.sukses else \
                       "DIAGNOSTIK SELESAI dengan PERINGATAN — Periksa log"
        self.root.after(0, self.set_status, teks_status, warna_status)
        self.root.after(0, self.lbl_live.config,
            {"text": "◉ SCAN SELESAI", "fg": "#f39c12"})
        self.root.after(0, self._aktifkan_tombol_launch)

    def _aktifkan_tombol_launch(self):
        self.btn_launch.config(
            text="🚀  Luncurkan Aplikasi Utama",
            bg="#27ae60", fg="white",
            state="normal", cursor="hand2",
            command=self._luncurkan_aplikasi)

    def _luncurkan_aplikasi(self):
        tulis_log("INFO", "Pengguna menekan LUNCURKAN — Membuka app.py...")
        self.root.destroy()
        os.execv(PYTHON, [PYTHON, APP_FILE])

    def jalankan(self):
        self.root.mainloop()


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    app = LauncherWindow()
    app.jalankan()
