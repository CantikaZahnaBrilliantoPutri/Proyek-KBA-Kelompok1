"""
Bronze Layer - Data Engineer Script
Proyek: Analisis Tren Pembelian Pelanggan Retail
Kelompok 1 - KBA-F, Universitas Brawijaya 2026
PIC: Azka Mitsalia Zamzami
 
Deskripsi:
    Script ini menangani tahap pertama dari pipeline ETL berbasis
    arsitektur Medallion. Bronze Layer bertugas memuat data mentah (raw)
    dari sumber CSV tanpa melakukan transformasi apapun.
    Seluruh kolom disimpan dalam tipe String untuk menjaga
    fidelitas data asli (data fidelity).
"""
 
import pandas as pd
import os
from datetime import datetime
 
# ==============================================================
# KONFIGURASI
# ==============================================================
CSV_FILE_PATH = "retail_store_transactions_23100.csv"   # Ganti path sesuai lokasi file CSV
OUTPUT_DIR    = "bronze_output"
OUTPUT_FILE   = os.path.join(OUTPUT_DIR, "bronze_transactions.csv")
LOG_FILE      = os.path.join(OUTPUT_DIR, "bronze_load_log.txt")
 
# Kolom yang diharapkan dari dataset Kaggle
EXPECTED_COLUMNS = [
    "Transaction_ID",
    "Customer_ID",
    "Age",
    "Gender",
    "Product_Category",
    "Quantity",
    "Price_Per_Unit",
    "Total_Purchase_Amount",
    "Payment_Method",
    "Transaction_Date",
    "Loyalty_Member",
    "Discount_Applied",
]
 
 
# ==============================================================
# FUNGSI UTILITAS
# ==============================================================
def log(message: str, log_lines: list):
    """Cetak dan simpan log ke dalam list."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    log_lines.append(line)
 
 
def save_log(log_lines: list):
    """Simpan log ke file teks."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"[LOG] Log tersimpan di: {LOG_FILE}")
 
 
# ==============================================================
# TAHAP 1: EXTRACT - Baca CSV
# ==============================================================
def extract(filepath: str, log_lines: list) -> pd.DataFrame:
    """
    Membaca file CSV dari sumber data (Kaggle dataset).
    Semua kolom dibaca sebagai String (dtype=str) untuk menjaga
    fidelitas data pada Bronze Layer — tidak ada konversi tipe data.
    """
    log(f"[EXTRACT] Membaca file: {filepath}", log_lines)
 
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"File tidak ditemukan: {filepath}\n"
            "Pastikan dataset sudah diunduh dari:\n"
            "https://www.kaggle.com/datasets/miadul/retail-store-transactions-dataset"
        )
 
    # Baca semua kolom sebagai String — Bronze Layer tidak melakukan casting
    df = pd.read_csv(filepath, dtype=str, keep_default_na=False)
 
    log(f"[EXTRACT] Berhasil dimuat: {len(df):,} baris, {len(df.columns)} kolom", log_lines)
    return df
 
 
# ==============================================================
# TAHAP 2: VALIDASI KOLOM (tanpa transformasi)
# ==============================================================
def validate_schema(df: pd.DataFrame, log_lines: list) -> pd.DataFrame:
    """
    Validasi keberadaan kolom yang diharapkan.
    Jika ada kolom yang hilang, dicatat ke log (tidak error).
    Jika ada kolom tambahan, tetap disimpan (Bronze menyimpan semua data).
    """
    log("[VALIDATE] Memeriksa skema kolom...", log_lines)
 
    existing_cols = set(df.columns.tolist())
    expected_cols = set(EXPECTED_COLUMNS)
 
    missing = expected_cols - existing_cols
    extra   = existing_cols - expected_cols
 
    if missing:
        log(f"[VALIDATE] ⚠ Kolom tidak ditemukan: {missing}", log_lines)
    else:
        log("[VALIDATE] ✓ Semua kolom yang diharapkan tersedia.", log_lines)
 
    if extra:
        log(f"[VALIDATE] INFO Kolom tambahan (tetap disimpan): {extra}", log_lines)
 
    # Statistik awal (raw, tanpa transformasi)
    total_rows = len(df)
    empty_cells = (df == "").sum().sum()
    null_cells  = df.isnull().sum().sum()
 
    log(f"[VALIDATE] Total baris     : {total_rows:,}", log_lines)
    log(f"[VALIDATE] Sel kosong ('')  : {empty_cells:,}", log_lines)
    log(f"[VALIDATE] Sel null/NaN    : {null_cells:,}", log_lines)
    log("[VALIDATE] Catatan: Bronze Layer TIDAK menghapus/mengubah data apapun.", log_lines)
 
    return df
 
 
# ==============================================================
# TAHAP 3: LOAD - Simpan Bronze Output
# ==============================================================
def load(df: pd.DataFrame, log_lines: list):
    """
    Menyimpan data mentah ke Bronze output directory.
    Menambahkan kolom metadata: bronze_loaded_at dan bronze_source_file.
    Ini adalah satu-satunya penambahan yang dilakukan di Bronze Layer.
    """
    log("[LOAD] Menyimpan data ke Bronze output...", log_lines)
 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
 
    # Tambahkan metadata Bronze (satu-satunya modifikasi yang diperbolehkan)
    df["bronze_loaded_at"]    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df["bronze_source_file"]  = os.path.basename(CSV_FILE_PATH)
 
    df.to_csv(OUTPUT_FILE, index=False)
 
    log(f"[LOAD] ✓ Data Bronze tersimpan: {OUTPUT_FILE}", log_lines)
    log(f"[LOAD] Total baris tersimpan  : {len(df):,}", log_lines)
    log(f"[LOAD] Total kolom            : {len(df.columns)}", log_lines)
 
 
# ==============================================================
# MAIN PIPELINE
# ==============================================================
def run_bronze_pipeline():
    """Menjalankan pipeline Bronze Layer secara berurutan."""
    log_lines = []
    log("=" * 60, log_lines)
    log("  BRONZE LAYER PIPELINE - START", log_lines)
    log("  Proyek: Analisis Tren Pembelian Pelanggan Retail", log_lines)
    log("  PIC   : Azka Mitsalia Zamzami", log_lines)
    log("=" * 60, log_lines)
 
    try:
        # Step 1: Extract
        df = extract(CSV_FILE_PATH, log_lines)
 
        # Step 2: Validate (no transformation)
        df = validate_schema(df, log_lines)
 
        # Step 3: Load
        load(df, log_lines)
 
        log("=" * 60, log_lines)
        log("  BRONZE LAYER PIPELINE - SELESAI ✓", log_lines)
        log("=" * 60, log_lines)
 
    except FileNotFoundError as e:
        log(f"[ERROR] {e}", log_lines)
    except Exception as e:
        log(f"[ERROR] Terjadi kesalahan: {e}", log_lines)
    finally:
        save_log(log_lines)
 
 
if __name__ == "__main__":
    run_bronze_pipeline()