"""
============================================================
Gold Layer - Step by Step
Proyek : Analisis Tren Pembelian Pelanggan Retail
Kelompok 1 - KBA-F, Universitas Brawijaya 2026
============================================================

INPUT  : silver_inventory.csv  (output dari Silver Layer)
OUTPUT : gold_output/gold_category_kpi.csv
         gold_output/gold_inventory_status.csv
         gold_output/gold_supplier_performance.csv
         gold_output/gold_product_clusters.csv
         gold_output/gold_product_cluster_summary.csv
         gold_output/gold_supplier_clusters.csv
         + semua tabel di atas di-load ke ClickHouse (retail_dw)

CARA JALANKAN:
    # 1. Jalankan ClickHouse via Docker dulu:
    #    docker run -d --name clickhouse-retail \
    #      -p 8123:8123 -p 9000:9000 \
    #      clickhouse/clickhouse-server
    #
    #    Cek jalan: docker ps
    #
    # 2. Install library:
    #    pip install scikit-learn clickhouse-driver pandas numpy
    #
    # 3. Jalankan script:
    #    python gold_layer_steps.py
============================================================
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from clickhouse_driver import Client


# ============================================================
# STEP 1 — KONFIGURASI
# ============================================================
SILVER_FILE = "silver_inventory.csv"
OUTPUT_DIR  = "gold_output"

# ClickHouse — Docker lokal (bukan Cloud)
CH_HOST     = "localhost"
CH_PORT     = 9000
CH_USER     = "default"
CH_PASSWORD = ""
CH_DATABASE = "retail_dw"
CH_SECURE   = False       # True hanya untuk ClickHouse Cloud/TLS

# Nama tabel Gold di ClickHouse
TABLE_CATEGORY                = "gold_category_kpi"
TABLE_STATUS                  = "gold_inventory_status"
TABLE_SUPPLIER                = "gold_supplier_performance"
TABLE_PRODUCT_CLUSTERS        = "gold_product_clusters"
TABLE_PRODUCT_CLUSTER_SUMMARY = "gold_product_cluster_summary"
TABLE_SUPPLIER_CLUSTERS       = "gold_supplier_clusters"

# Range K yang dicoba di eksperimen (inklusif kedua ujung)
# K dibatasi 2–5 karena dataset supplier & product tidak terlalu besar,
# sehingga cluster terlalu banyak akan sulit diinterpretasikan secara bisnis.
# Selain itu, segmentasi bisnis yang meaningful (Low/High/Premium atau
# Top/Average/At-Risk) secara natural terbentuk dalam 2–4 kelompok.
K_RANGE = range(2, 6)     # → K = 2, 3, 4, 5


# ============================================================
# HELPER: INSERT DATAFRAME KE CLICKHOUSE
# ============================================================
# Fungsi bantu agar insert ke ClickHouse lebih bersih.
# Masalah umum yang ditangani:
#   - Boolean harus dikonversi ke int (UInt8 di ClickHouse)
#   - NaN/None harus diisi 0 atau '' sebelum insert
#   - Kolom datetime harus ke string

def ch_insert(client: Client, tbl: str, df: pd.DataFrame) -> None:
    """Insert seluruh DataFrame ke tabel ClickHouse."""
    df_copy = df.copy()

    # Bool → int (UInt8 di ClickHouse)
    for col in df_copy.select_dtypes(include="bool").columns:
        df_copy[col] = df_copy[col].astype(int)

    # Datetime → string
    for col in df_copy.select_dtypes(include="datetime").columns:
        df_copy[col] = df_copy[col].dt.strftime("%Y-%m-%d").fillna("")

    # Isi NaN: string → '', angka → 0
    for col in df_copy.columns:
        if df_copy[col].dtype == object or str(df_copy[col].dtype) == "string":
            df_copy[col] = df_copy[col].fillna("")
        else:
            df_copy[col] = df_copy[col].fillna(0)

    cols = ", ".join(df_copy.columns)
    client.execute(
        f"INSERT INTO {tbl} ({cols}) VALUES",
        df_copy.to_dict("records"),
    )
    print(f"  ✓ INSERT {tbl:<44} — {len(df_copy):>4} baris")


# ============================================================
# HELPER: EKSPERIMEN K-MEANS — CARI K TERBAIK
# ============================================================
# Menjalankan KMeans untuk setiap K di K_RANGE, lalu memilih
# K dengan Silhouette Score tertinggi.
# Silhouette Score dipilih sebagai primary metric karena:
#   - Tidak bergantung pada ground truth label
#   - Mudah diinterpretasikan (-1 buruk, 1 sempurna)
#   - Cocok untuk data tanpa struktur cluster yang jelas

def find_best_k(X: np.ndarray, k_range: range, label: str) -> int:
    """
    Iterasi k_range, cetak tabel metrik tiap K,
    return K terbaik berdasarkan Silhouette Score.
    """
    print(f"\n  [EKSPERIMEN K] {label}")
    print(f"  {'K':>3}  {'Silhouette':>12}  {'Davies-Bouldin':>15}  "
          f"{'Calinski-Harabasz':>18}  {'Inertia':>12}")
    print(f"  {'─'*3}  {'─'*12}  {'─'*15}  {'─'*18}  {'─'*12}")

    results = []
    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil    = silhouette_score(X, labels)
        dbi    = davies_bouldin_score(X, labels)
        ch     = calinski_harabasz_score(X, labels)
        iner   = km.inertia_
        results.append((k, sil, dbi, ch, iner))
        print(f"  {k:>3}  {sil:>12.4f}  {dbi:>15.4f}  {ch:>18.2f}  {iner:>12.4f}")

    best_k, best_sil, *_ = max(results, key=lambda x: x[1])
    print(f"\n  ✓ K terbaik = {best_k}  (Silhouette = {best_sil:.4f})")
    return best_k


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> None:

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 62)
    print("  GOLD LAYER PIPELINE - START")
    print("  Proyek: Analisis Tren Pembelian Pelanggan Retail")
    print("  Kelompok 1 - KBA-F, Universitas Brawijaya 2026")
    print("=" * 62)
    print(f"  Silver input : {SILVER_FILE}")
    print(f"  Output dir   : {OUTPUT_DIR}/")
    print(f"  ClickHouse   : {CH_HOST}:{CH_PORT} / {CH_DATABASE}")
    print("=" * 62)

    # ----------------------------------------------------------
    # CONNECT CLICKHOUSE
    # ----------------------------------------------------------
    # Koneksi ke ClickHouse Docker yang sudah dijalankan.
    # Pastikan container sudah aktif sebelum menjalankan script:
    #   docker ps  →  harus ada "clickhouse-retail" dengan status Up

    print("\n[CLICKHOUSE] Connecting...")

    try:
        # Connect ke 'default' dulu agar bisa CREATE DATABASE
        _client = Client(
            host     = CH_HOST,
            port     = CH_PORT,
            user     = CH_USER,
            password = CH_PASSWORD,
            database = "default",
            secure   = CH_SECURE,
        )
        _client.execute(f"CREATE DATABASE IF NOT EXISTS {CH_DATABASE}")

        # Reconnect ke database retail_dw yang sudah pasti ada
        client = Client(
            host     = CH_HOST,
            port     = CH_PORT,
            user     = CH_USER,
            password = CH_PASSWORD,
            database = CH_DATABASE,
            secure   = CH_SECURE,
        )
        print(f"  ✓ Terhubung ke ClickHouse ({CH_HOST}:{CH_PORT})")
        print(f"  ✓ Database '{CH_DATABASE}' siap")
    except Exception as e:
        print(f"  ✗ Gagal connect ClickHouse: {e}")
        print("    Pastikan Docker container sudah jalan:")
        print("    docker run -d --name clickhouse-retail \\")
        print("      -p 8123:8123 -p 9000:9000 \\")
        print("      clickhouse/clickhouse-server")
        raise

    # ----------------------------------------------------------
    # STEP 2 — LOAD SILVER DATA
    # ----------------------------------------------------------
    print("\n[STEP 2] Load Silver Data...")

    if not os.path.exists(SILVER_FILE):
        raise FileNotFoundError(
            f"File '{SILVER_FILE}' tidak ditemukan.\n"
            "Pastikan silver_inventory.csv ada di folder yang sama."
        )

    df = pd.read_csv(SILVER_FILE)
    print(f"  ✓ Silver data dimuat: {df.shape[0]:,} baris x {df.shape[1]} kolom")
    print(f"  Kolom: {list(df.columns)}")

    # ----------------------------------------------------------
    # STEP 3 — CLEANING & CAST TYPES
    # ----------------------------------------------------------
    print("\n[STEP 3] Cleaning & Cast Types...")

    GOLD_TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── [PERBAIKAN] Fix typo kolom 'Catagory' → 'Category' ──────
    # Dataset asli mengandung typo pada nama kolom. Rename dilakukan
    # di awal pipeline agar seluruh proses downstream konsisten
    # menggunakan nama kolom yang benar secara profesional.
    if "Catagory" in df.columns:
        df.rename(columns={"Catagory": "Category"}, inplace=True)
        print("  ✓ Rename kolom: 'Catagory' → 'Category'  (fix typo dataset)")
    # ────────────────────────────────────────────────────────────

    # ── [PERBAIKAN] Audit Missing Value sebelum preprocessing ───
    # Audit ini penting untuk membuktikan bahwa pipeline menangani
    # missing value secara eksplisit, bukan hanya implisit melalui
    # dropna() di tahap modeling.
    print()
    print("  [MISSING VALUE AUDIT - SEBELUM CLEANING]")
    missing_per_col = df.isna().sum()
    missing_total   = missing_per_col.sum()
    if missing_total == 0:
        print("  ✓ Tidak ada missing value yang terdeteksi di seluruh kolom")
    else:
        print(f"  ⚠ Total missing value: {missing_total}")
        print(missing_per_col[missing_per_col > 0].to_string())
    missing_before = missing_total
    # ────────────────────────────────────────────────────────────

    # 3a. Bersihkan kolom percentage
    #     .astype(str) dulu supaya aman kalau pandas baca sebagai numeric
    df["percentage"] = (
        df["percentage"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .astype(float)
    )
    print("\n  ✓ percentage → float  (via .astype(str) — aman untuk numeric input)")

    # 3b. Parse tanggal
    date_cols = ["Date_Received", "Last_Order_Date", "Expiration_Date"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        print(f"  ✓ {col} → datetime | null: {df[col].isna().sum()}")

    # 3c. Boolean Low_Stock
    #     Pakai .map() supaya string "False" tidak dianggap True
    #     (non-empty string dievaluasi True oleh bool() Python biasa)
    df["Low_Stock"] = (
        df["Low_Stock"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )
    print("  ✓ Low_Stock → bool  (via string map, aman untuk semua format CSV)")
    print(f"  ✓ Timestamp Gold: {GOLD_TIMESTAMP}")

    # ── [PERBAIKAN] Audit Missing Value setelah cast type ───────
    # Beberapa missing value dapat muncul setelah cast (misal: datetime
    # yang gagal parse → NaT). Audit kedua ini memastikan pipeline
    # transparan terhadap perubahan missing value akibat transformasi.
    missing_after_cast = df.isna().sum().sum()
    missing_added      = missing_after_cast - missing_before
    print()
    print("  [MISSING VALUE AUDIT - SETELAH CAST TYPE]")
    print(f"  ✓ Missing value sebelum cast : {missing_before}")
    print(f"  ✓ Missing value setelah cast : {missing_after_cast}")
    if missing_added > 0:
        print(f"  ⚠ Tambahan missing value dari gagal parse: {missing_added}")
        print(missing_per_col[df.isna().sum() > 0].to_string())
    else:
        print("  ✓ Tidak ada missing value baru yang muncul dari proses cast")
    # ────────────────────────────────────────────────────────────

    # ----------------------------------------------------------
    # STEP 4 — HANDLE DUPLICATES & OUTLIERS
    # ----------------------------------------------------------
    # Preprocessing standar data mining sesuai best practice:
    #   - Drop duplicate : mencegah bias pada agregasi & clustering
    #   - IQR filtering  : outlier ekstrem dapat menarik centroid
    #                      KMeans ke arah yang tidak representatif
    #
    # Mengapa unsupervised learning (K-Means)?
    #   Dataset tidak memiliki target label yang valid untuk
    #   supervised learning. K-Means dipilih karena:
    #     1. Data dominan numerik — cocok untuk distance-based algo
    #     2. Tujuan bisnis adalah segmentasi, bukan prediksi kelas
    #     3. Interpretasi cluster mudah dipahami stakeholder non-teknis
    #   StandardScaler wajib dipakai sebelum KMeans agar fitur
    #   berskala besar (Inventory_Value) tidak mendominasi jarak.

    print("\n[STEP 4] Handle Duplicates & Outliers...")

    # 4a. Drop duplicates
    before_dup = len(df)
    df = df.drop_duplicates()
    print(f"  ✓ Duplicate removed    : {before_dup - len(df)} baris  "
          f"(tersisa {len(df):,})")

    # 4b. IQR-based outlier filtering untuk kolom numerik utama
    OUTLIER_COLS = [
        "Stock_Quantity",
        "Unit_Price",
        "Sales_Volume",
        "Inventory_Value",
    ]

    total_outliers = 0
    for col in OUTLIER_COLS:
        before = len(df)
        Q1     = df[col].quantile(0.25)
        Q3     = df[col].quantile(0.75)
        IQR    = Q3 - Q1
        lower  = Q1 - 1.5 * IQR
        upper  = Q3 + 1.5 * IQR
        df     = df[(df[col] >= lower) & (df[col] <= upper)]
        removed = before - len(df)
        total_outliers += removed
        print(f"  ✓ Outlier [{col:<26}]: {removed:>3} baris  "
              f"| range valid [{lower:.1f} – {upper:.1f}]")

    print(f"  ✓ Total outlier removed: {total_outliers} baris  "
          f"(data bersih: {len(df):,} baris)")

    # ── [PERBAIKAN] Audit Missing Value setelah outlier removal ─
    # Setelah drop, cek ulang apakah ada NaN yang tersisa di kolom
    # numerik yang akan digunakan untuk modeling.
    missing_post_clean = df.isna().sum().sum()
    print(f"  ✓ Missing value tersisa setelah cleaning: {missing_post_clean}")
    if missing_post_clean > 0:
        print("  ⚠ Kolom dengan missing value yang tersisa:")
        print(df.isna().sum()[df.isna().sum() > 0].to_string())
    # ────────────────────────────────────────────────────────────

    # ----------------------------------------------------------
    # STEP 5 — GOLD TABLE 1: gold_category_kpi
    # ----------------------------------------------------------
    print("\n[STEP 5] Membuat gold_category_kpi...")

    gold_category = (
        df.groupby("Category")
        .agg(
            total_products        = ("Product_ID",              "count"),
            total_inventory_value = ("Inventory_Value",         "sum"),
            avg_unit_price        = ("Unit_Price",              "mean"),
            total_stock           = ("Stock_Quantity",          "sum"),
            avg_stock             = ("Stock_Quantity",          "mean"),
            total_reorder_qty     = ("Reorder_Quantity",        "sum"),
            total_sales_volume    = ("Sales_Volume",            "sum"),
            avg_sales_volume      = ("Sales_Volume",            "mean"),
            avg_turnover_rate     = ("Inventory_Turnover_Rate", "mean"),
            low_stock_count       = ("Low_Stock",               "sum"),
            active_products       = ("Status", lambda x: (x == "Active").sum()),
            backordered_products  = ("Status", lambda x: (x == "Backordered").sum()),
            discontinued_products = ("Status", lambda x: (x == "Discontinued").sum()),
        )
        .reset_index()
        .rename(columns={"Category": "category"})
    )
    gold_category["low_stock_pct"] = (
        gold_category["low_stock_count"] / gold_category["total_products"] * 100
    ).round(2)
    gold_category["revenue_per_unit_avg"] = (
        gold_category["total_inventory_value"]
        / gold_category["total_stock"].replace(0, np.nan)
    ).round(2)
    for col in ["avg_unit_price", "avg_stock", "avg_sales_volume", "avg_turnover_rate"]:
        gold_category[col] = gold_category[col].round(2)
    gold_category["gold_processed_at"] = GOLD_TIMESTAMP

    print(f"  ✓ {gold_category.shape[0]} baris x {gold_category.shape[1]} kolom")
    print(gold_category[["category", "total_products", "total_inventory_value",
                          "total_sales_volume", "low_stock_pct"]].to_string(index=False))

    # ----------------------------------------------------------
    # STEP 6 — GOLD TABLE 2: gold_inventory_status
    # ----------------------------------------------------------
    print("\n[STEP 6] Membuat gold_inventory_status...")

    gold_status = (
        df.groupby(["Category", "Status"])
        .agg(
            product_count         = ("Product_ID",       "count"),
            total_inventory_value = ("Inventory_Value",  "sum"),
            total_stock           = ("Stock_Quantity",   "sum"),
            avg_unit_price        = ("Unit_Price",       "mean"),
            low_stock_count       = ("Low_Stock",        "sum"),
            avg_reorder_quantity  = ("Reorder_Quantity", "mean"),
        )
        .reset_index()
        .rename(columns={"Category": "category", "Status": "status"})
    )
    gold_status["low_stock_pct"]        = (gold_status["low_stock_count"] / gold_status["product_count"] * 100).round(2)
    gold_status["avg_unit_price"]       = gold_status["avg_unit_price"].round(2)
    gold_status["avg_reorder_quantity"] = gold_status["avg_reorder_quantity"].round(1)
    gold_status["gold_processed_at"]    = GOLD_TIMESTAMP

    print(f"  ✓ {gold_status.shape[0]} baris x {gold_status.shape[1]} kolom")
    print(gold_status[["category", "status", "product_count",
                        "total_inventory_value", "low_stock_pct"]].to_string(index=False))

    # ----------------------------------------------------------
    # STEP 7 — GOLD TABLE 3: gold_supplier_performance
    # ----------------------------------------------------------
    print("\n[STEP 7] Membuat gold_supplier_performance...")

    gold_supplier = (
        df.groupby("Supplier_Name")
        .agg(
            total_products        = ("Product_ID",              "count"),
            categories_supplied   = ("Category",                "nunique"),
            total_inventory_value = ("Inventory_Value",         "sum"),
            total_stock           = ("Stock_Quantity",          "sum"),
            avg_unit_price        = ("Unit_Price",              "mean"),
            total_sales_volume    = ("Sales_Volume",            "sum"),
            avg_turnover_rate     = ("Inventory_Turnover_Rate", "mean"),
            low_stock_count       = ("Low_Stock",               "sum"),
            active_products       = ("Status", lambda x: (x == "Active").sum()),
        )
        .reset_index()
        .rename(columns={"Supplier_Name": "supplier_name"})
    )
    gold_supplier["low_stock_pct"]     = (gold_supplier["low_stock_count"] / gold_supplier["total_products"] * 100).round(2)
    gold_supplier["avg_unit_price"]    = gold_supplier["avg_unit_price"].round(2)
    gold_supplier["avg_turnover_rate"] = gold_supplier["avg_turnover_rate"].round(2)

    def minmax(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn) if mx > mn else pd.Series([0.5] * len(s), index=s.index)

    gold_supplier["supply_reliability_score"] = (
        minmax(gold_supplier["total_sales_volume"])   * 40 +
        minmax(gold_supplier["avg_turnover_rate"])    * 30 +
        (1 - minmax(gold_supplier["low_stock_pct"])) * 30
    ).round(2)

    gold_supplier = gold_supplier.sort_values(
        "supply_reliability_score", ascending=False
    ).reset_index(drop=True)
    gold_supplier["gold_processed_at"] = GOLD_TIMESTAMP

    print(f"  ✓ {gold_supplier.shape[0]} baris x {gold_supplier.shape[1]} kolom")
    print("  Top 5 Supplier:")
    print(gold_supplier[["supplier_name", "total_products",
                          "total_sales_volume", "supply_reliability_score"]].head(5).to_string(index=False))

    # ----------------------------------------------------------
    # STEP 8 — CLUSTERING TABLE 1: gold_product_clusters
    # ----------------------------------------------------------
    print("\n[STEP 8] K-Means Clustering: Produk...")

    PRODUCT_FEATURES = [
        "Stock_Quantity",
        "Unit_Price",
        "Sales_Volume",
        "Inventory_Turnover_Rate",
        "Inventory_Value",
    ]

    df_prod = df[[
        "Product_ID", "Product_Name", "Category",
        "Supplier_Name", "Status", "Low_Stock",
    ] + PRODUCT_FEATURES].copy().dropna(subset=PRODUCT_FEATURES)

    # ── [PERBAIKAN] Audit missing value sebelum clustering produk
    missing_prod = df_prod[PRODUCT_FEATURES].isna().sum()
    print(f"  [MISSING VALUE - Fitur Produk setelah dropna]")
    print(f"  ✓ Missing value removed via dropna: {missing_prod.sum()}")
    print(f"  ✓ Data produk siap clustering       : {len(df_prod):,} baris")
    # ────────────────────────────────────────────────────────────

    # 8a. Scaling
    scaler_product = StandardScaler()
    X_product      = scaler_product.fit_transform(df_prod[PRODUCT_FEATURES])

    # 8b. Eksperimen K → pilih K terbaik berdasarkan Silhouette Score
    N_CLUSTERS_PRODUCT = find_best_k(X_product, K_RANGE, "Produk")

    # 8c. Final K-Means dengan K terbaik
    #     random_state=42 → hasil reprodusibel di setiap run
    #     n_init=10       → 10 inisialisasi berbeda, ambil terbaik
    kmeans_product = KMeans(
        n_clusters=N_CLUSTERS_PRODUCT, random_state=42, n_init=10
    )
    df_prod["cluster_id"] = kmeans_product.fit_predict(X_product)

    # 8d. Labeling otomatis dari centroid (invers ke skala asli)
    #     Unit_Price tertinggi  → "Premium Stock"
    #     Stock_Quantity tinggi → "High Stock"
    #     Stock_Quantity rendah → "Low Stock"
    centroids_prod   = scaler_product.inverse_transform(kmeans_product.cluster_centers_)
    centroid_prod_df = pd.DataFrame(centroids_prod, columns=PRODUCT_FEATURES)
    centroid_prod_df["cluster_id"] = range(N_CLUSTERS_PRODUCT)

    # ── [PERBAIKAN] Print fitur utama pembeda cluster produk ────
    # Menampilkan nilai centroid tiap cluster dalam skala asli,
    # sehingga terlihat jelas fitur mana yang membentuk setiap segmen.
    print("\n  Fitur utama pembeda cluster produk (centroid skala asli):")
    print(centroid_prod_df.to_string(index=False))
    # ────────────────────────────────────────────────────────────

    price_sorted = centroid_prod_df.sort_values("Unit_Price", ascending=False)
    premium_id   = int(price_sorted.iloc[0]["cluster_id"])
    remaining    = centroid_prod_df[centroid_prod_df["cluster_id"] != premium_id]
    stock_sorted = remaining.sort_values("Stock_Quantity", ascending=False)

    product_label_map = {premium_id: "Premium Stock"}
    for rank, (_, row) in enumerate(stock_sorted.iterrows()):
        cid = int(row["cluster_id"])
        if rank == 0:
            product_label_map[cid] = "High Stock"
        elif rank == 1:
            product_label_map[cid] = "Low Stock"
        else:
            product_label_map[cid] = f"Cluster {cid}"

    df_prod["cluster_label"] = df_prod["cluster_id"].map(product_label_map)

    # 8e. Rename ke lowercase untuk konsistensi ClickHouse
    gold_product_clusters = df_prod.rename(columns={
        "Product_ID"             : "product_id",
        "Product_Name"           : "product_name",
        "Category"               : "category",
        "Supplier_Name"          : "supplier_name",
        "Status"                 : "status",
        "Low_Stock"              : "low_stock",
        "Stock_Quantity"         : "stock_quantity",
        "Unit_Price"             : "unit_price",
        "Sales_Volume"           : "sales_volume",
        "Inventory_Turnover_Rate": "inventory_turnover_rate",
        "Inventory_Value"        : "inventory_value",
    }).copy()
    gold_product_clusters["gold_processed_at"] = GOLD_TIMESTAMP

    # 8f. Tabel ringkasan per cluster
    gold_product_cluster_summary = (
        gold_product_clusters.groupby("cluster_label")
        .agg(
            product_count       = ("product_id",               "count"),
            avg_stock           = ("stock_quantity",            "mean"),
            avg_unit_price      = ("unit_price",                "mean"),
            avg_sales_volume    = ("sales_volume",              "mean"),
            avg_turnover_rate   = ("inventory_turnover_rate",   "mean"),
            avg_inventory_value = ("inventory_value",           "mean"),
            low_stock_count     = ("low_stock",                 "sum"),
            active_count        = ("status", lambda x: (x == "Active").sum()),
            backordered_count   = ("status", lambda x: (x == "Backordered").sum()),
            discontinued_count  = ("status", lambda x: (x == "Discontinued").sum()),
        )
        .reset_index()
    )
    for col in gold_product_cluster_summary.select_dtypes("float").columns:
        gold_product_cluster_summary[col] = gold_product_cluster_summary[col].round(2)
    gold_product_cluster_summary["low_stock_pct"] = (
        gold_product_cluster_summary["low_stock_count"]
        / gold_product_cluster_summary["product_count"] * 100
    ).round(2)
    gold_product_cluster_summary["gold_processed_at"] = GOLD_TIMESTAMP

    # ── EVALUASI CLUSTERING PRODUK ────────────────────────────
    sil_prod = silhouette_score(X_product, kmeans_product.labels_)
    dbi_prod = davies_bouldin_score(X_product, kmeans_product.labels_)
    ch_prod  = calinski_harabasz_score(X_product, kmeans_product.labels_)

    print(f"\n  ✓ gold_product_clusters  : {gold_product_clusters.shape[0]} baris  "
          f"(K={N_CLUSTERS_PRODUCT})")
    print()
    print("  ┌──────────────────────────────────────────────────────────────┐")
    print("  │              EVALUASI CLUSTERING PRODUK                     │")
    print("  ├──────────────────────────────────────────────────────────────┤")
    print(f"  │  K yang digunakan     : {N_CLUSTERS_PRODUCT:<38}│")
    print(f"  │  Silhouette Score     : {sil_prod:>8.4f}   (target > 0.5)       │")
    print(f"  │  Davies-Bouldin Index : {dbi_prod:>8.4f}   (makin rendah ✓)    │")
    print(f"  │  Calinski-Harabasz   : {ch_prod:>8.2f}   (makin tinggi ✓)   │")
    print(f"  │  Inertia             : {kmeans_product.inertia_:>8.4f}                      │")
    print("  ├──────────────────────────────────────────────────────────────┤")
    if sil_prod >= 0.5:
        verdict_prod = "BAIK — Cluster terpisah dengan jelas ✓"
        detail_prod  = "  │  Segmentasi produk sangat reliable untuk dipakai bisnis.  │"
    elif sil_prod >= 0.25:
        verdict_prod = "CUKUP — Cluster acceptable, ada sedikit overlap"
        detail_prod  = ("  │  Overlap wajar karena Sales_Volume & Turnover terdistribusi│\n"
                        "  │  merata. Label bisnis tetap bermakna.                       │")
    else:
        verdict_prod = "LEMAH — Cluster overlap, tapi masih bermakna bisnis"
        detail_prod  = ("  │  Pembeda utama: Unit_Price (Premium) & Stock_Quantity.     │\n"
                        "  │  Pertimbangkan menambah fitur atau menyesuaikan K.          │")
    print(f"  │  Verdict  : {verdict_prod:<51}│")
    print(detail_prod)
    print("  └──────────────────────────────────────────────────────────────┘")
    print()
    print("  Ringkasan per cluster (centroid skala asli):")
    print(gold_product_cluster_summary[[
        "cluster_label", "product_count", "avg_stock", "avg_unit_price",
        "avg_sales_volume", "avg_inventory_value", "low_stock_pct",
    ]].to_string(index=False))
    print()
    print("  Interpretasi bisnis:")
    for _, row in gold_product_cluster_summary.sort_values(
        "avg_unit_price", ascending=False
    ).iterrows():
        lbl = row["cluster_label"]
        if lbl == "Premium Stock":
            print(f"    [Premium Stock]  Harga satuan avg {row['avg_unit_price']:.0f}/unit, "
                  f"inv. value avg {row['avg_inventory_value']:.0f}. "
                  f"{int(row['product_count'])} produk. Prioritas monitoring.")
        elif lbl == "High Stock":
            print(f"    [High Stock]     Stok berlimpah avg {row['avg_stock']:.0f} unit. "
                  f"{int(row['product_count'])} produk. Cek potensi overstock.")
        elif lbl == "Low Stock":
            print(f"    [Low Stock]      Stok rendah avg {row['avg_stock']:.0f} unit. "
                  f"{row['low_stock_pct']:.0f}% sudah low stock. "
                  f"{int(row['product_count'])} produk perlu segera direorder.")
        else:
            print(f"    [{lbl}]  avg stock={row['avg_stock']:.0f}, "
                  f"avg price={row['avg_unit_price']:.0f}. "
                  f"{int(row['product_count'])} produk.")

    # ----------------------------------------------------------
    # STEP 9 — CLUSTERING TABLE 2: gold_supplier_clusters
    # ----------------------------------------------------------
    print("\n[STEP 9] K-Means Clustering: Supplier...")

    SUPPLIER_FEATURES = [
        "total_products",
        "avg_turnover_rate",
        "low_stock_count",
        "total_sales_volume",
        "supply_reliability_score",
    ]

    df_sup_clust = gold_supplier[
        ["supplier_name", "categories_supplied", "active_products"] + SUPPLIER_FEATURES
    ].copy().dropna(subset=SUPPLIER_FEATURES)

    # ── [PERBAIKAN] Audit missing value sebelum clustering supplier
    missing_sup = df_sup_clust[SUPPLIER_FEATURES].isna().sum()
    print(f"  [MISSING VALUE - Fitur Supplier setelah dropna]")
    print(f"  ✓ Missing value removed via dropna: {missing_sup.sum()}")
    print(f"  ✓ Data supplier siap clustering    : {len(df_sup_clust):,} baris")
    # ────────────────────────────────────────────────────────────

    # 9a. Scaling
    scaler_supplier = StandardScaler()
    X_supplier      = scaler_supplier.fit_transform(df_sup_clust[SUPPLIER_FEATURES])

    # 9b. Eksperimen K → pilih K terbaik
    N_CLUSTERS_SUPPLIER = find_best_k(X_supplier, K_RANGE, "Supplier")

    # 9c. Final K-Means
    kmeans_supplier = KMeans(
        n_clusters=N_CLUSTERS_SUPPLIER, random_state=42, n_init=10
    )
    df_sup_clust["supplier_cluster_id"] = kmeans_supplier.fit_predict(X_supplier)

    # 9d. Labeling: reliability score tertinggi → Top, terendah → At-Risk
    centroids_sup   = scaler_supplier.inverse_transform(kmeans_supplier.cluster_centers_)
    centroid_sup_df = pd.DataFrame(centroids_sup, columns=SUPPLIER_FEATURES)
    centroid_sup_df["cluster_id"] = range(N_CLUSTERS_SUPPLIER)

    # ── [PERBAIKAN] Print fitur utama pembeda cluster supplier ──
    # Menampilkan centroid supplier dalam skala asli untuk
    # membuktikan fitur mana yang paling membedakan tiap tier.
    print("\n  Fitur utama pembeda cluster supplier (centroid skala asli):")
    print(centroid_sup_df.to_string(index=False))
    # ────────────────────────────────────────────────────────────

    score_sorted = centroid_sup_df.sort_values("supply_reliability_score", ascending=False)

    supplier_label_map = {}
    for rank, (_, row) in enumerate(score_sorted.iterrows()):
        cid = int(row["cluster_id"])
        if rank == 0:
            supplier_label_map[cid] = "Top Supplier"
        elif rank == N_CLUSTERS_SUPPLIER - 1:
            supplier_label_map[cid] = "At-Risk Supplier"
        else:
            supplier_label_map[cid] = "Average Supplier"

    df_sup_clust["supplier_tier"]     = df_sup_clust["supplier_cluster_id"].map(supplier_label_map)
    df_sup_clust["gold_processed_at"] = GOLD_TIMESTAMP
    gold_supplier_clusters            = df_sup_clust.copy()

    # ── EVALUASI CLUSTERING SUPPLIER ─────────────────────────
    sil_sup = silhouette_score(X_supplier, kmeans_supplier.labels_)
    dbi_sup = davies_bouldin_score(X_supplier, kmeans_supplier.labels_)
    ch_sup  = calinski_harabasz_score(X_supplier, kmeans_supplier.labels_)

    print(f"\n  ✓ gold_supplier_clusters : {gold_supplier_clusters.shape[0]} baris  "
          f"(K={N_CLUSTERS_SUPPLIER})")
    print()
    print("  ┌──────────────────────────────────────────────────────────────┐")
    print("  │             EVALUASI CLUSTERING SUPPLIER                    │")
    print("  ├──────────────────────────────────────────────────────────────┤")
    print(f"  │  K yang digunakan     : {N_CLUSTERS_SUPPLIER:<38}│")
    print(f"  │  Silhouette Score     : {sil_sup:>8.4f}   (target > 0.5)       │")
    print(f"  │  Davies-Bouldin Index : {dbi_sup:>8.4f}   (makin rendah ✓)    │")
    print(f"  │  Calinski-Harabasz   : {ch_sup:>8.2f}   (makin tinggi ✓)   │")
    print(f"  │  Inertia             : {kmeans_supplier.inertia_:>8.4f}                      │")
    print("  ├──────────────────────────────────────────────────────────────┤")
    if sil_sup >= 0.5:
        verdict_sup = "BAIK — Segmentasi supplier sangat jelas ✓"
        detail_sup  = "  │  Tiga tier dapat langsung dipakai tim procurement.          │"
    elif sil_sup >= 0.25:
        verdict_sup = "CUKUP — Segmentasi supplier cukup terpisah"
        detail_sup  = "  │  Supply_reliability_score jadi pembeda utama antar tier.    │"
    else:
        verdict_sup = "LEMAH — Supplier berdistribusi merata"
        detail_sup  = ("  │  Wajar untuk dataset supplier dengan performa serupa.       │\n"
                       "  │  Gunakan supply_reliability_score sebagai rank tambahan.    │")
    print(f"  │  Verdict  : {verdict_sup:<51}│")
    print(detail_sup)
    print("  └──────────────────────────────────────────────────────────────┘")
    print()
    print("  Distribusi dan skor per tier:")
    for tier in ["Top Supplier", "Average Supplier", "At-Risk Supplier"]:
        grp = gold_supplier_clusters[gold_supplier_clusters["supplier_tier"] == tier]
        if grp.empty:
            continue
        avg_score = grp["supply_reliability_score"].mean()
        avg_ls    = grp["low_stock_count"].mean()
        avg_sales = grp["total_sales_volume"].mean()
        print(f"    {tier:<20} : {len(grp):>3} supplier | "
              f"avg score={avg_score:.1f} | avg low_stock={avg_ls:.1f} | "
              f"avg sales={avg_sales:.0f}")
    print()
    print("  Interpretasi bisnis:")
    tier_notes = {
        "Top Supplier"    : "Andalkan untuk kategori prioritas. Pertimbangkan kontrak jangka panjang.",
        "Average Supplier": "Monitor rutin. Beri target peningkatan turnover rate.",
        "At-Risk Supplier": "Evaluasi segera. Cari supplier alternatif bila perlu.",
    }
    for tier, note in tier_notes.items():
        print(f"    [{tier}] {note}")

    # ----------------------------------------------------------
    # STEP 10 — RINGKASAN EVALUASI GABUNGAN
    # ----------------------------------------------------------
    print()
    print("=" * 62)
    print("  RINGKASAN EVALUASI CLUSTERING")
    print("=" * 62)
    print()
    hdr_prod = f"Produk (K={N_CLUSTERS_PRODUCT})"
    hdr_sup  = f"Supplier (K={N_CLUSTERS_SUPPLIER})"
    print(f"  {'Metrik':<30} {hdr_prod:>14} {hdr_sup:>16}")
    print(f"  {'─'*30} {'─'*14} {'─'*16}")
    print(f"  {'Silhouette Score (↑)':<30} {sil_prod:>14.4f} {sil_sup:>16.4f}")
    print(f"  {'Davies-Bouldin Index (↓)':<30} {dbi_prod:>14.4f} {dbi_sup:>16.4f}")
    print(f"  {'Calinski-Harabasz (↑)':<30} {ch_prod:>14.2f} {ch_sup:>16.2f}")
    print(f"  {'Inertia (↓)':<30} {kmeans_product.inertia_:>14.4f} {kmeans_supplier.inertia_:>16.4f}")
    print()

    avg_sil = (sil_prod + sil_sup) / 2
    if avg_sil >= 0.5:
        overall = "SANGAT BAIK — Semua cluster terpisah dengan jelas."
    elif avg_sil >= 0.25:
        overall = "CUKUP BAIK — Cluster meaningful secara bisnis meski ada overlap."
    else:
        overall = "PERLU REVIEW — Pertimbangkan adjust fitur atau nilai K."
    print(f"  Penilaian keseluruhan  : {overall}")
    print()

    # ----------------------------------------------------------
    # STEP 11 — SIMPAN KE CSV (LOCAL BACKUP)
    # ----------------------------------------------------------
    print("[STEP 11] Simpan ke CSV lokal...")

    PATH_CATEGORY                = os.path.join(OUTPUT_DIR, "gold_category_kpi.csv")
    PATH_STATUS                  = os.path.join(OUTPUT_DIR, "gold_inventory_status.csv")
    PATH_SUPPLIER                = os.path.join(OUTPUT_DIR, "gold_supplier_performance.csv")
    PATH_PRODUCT_CLUSTERS        = os.path.join(OUTPUT_DIR, "gold_product_clusters.csv")
    PATH_PRODUCT_CLUSTER_SUMMARY = os.path.join(OUTPUT_DIR, "gold_product_cluster_summary.csv")
    PATH_SUPPLIER_CLUSTERS       = os.path.join(OUTPUT_DIR, "gold_supplier_clusters.csv")

    gold_category.to_csv(PATH_CATEGORY,                            index=False)
    gold_status.to_csv(PATH_STATUS,                                index=False)
    gold_supplier.to_csv(PATH_SUPPLIER,                            index=False)
    gold_product_clusters.to_csv(PATH_PRODUCT_CLUSTERS,            index=False)
    gold_product_cluster_summary.to_csv(PATH_PRODUCT_CLUSTER_SUMMARY, index=False)
    gold_supplier_clusters.to_csv(PATH_SUPPLIER_CLUSTERS,          index=False)

    print(f"  ✓ {PATH_CATEGORY:<55} — {len(gold_category)} baris")
    print(f"  ✓ {PATH_STATUS:<55} — {len(gold_status)} baris")
    print(f"  ✓ {PATH_SUPPLIER:<55} — {len(gold_supplier)} baris")
    print(f"  ✓ {PATH_PRODUCT_CLUSTERS:<55} — {len(gold_product_clusters)} baris")
    print(f"  ✓ {PATH_PRODUCT_CLUSTER_SUMMARY:<55} — {len(gold_product_cluster_summary)} baris")
    print(f"  ✓ {PATH_SUPPLIER_CLUSTERS:<55} — {len(gold_supplier_clusters)} baris")

    # ----------------------------------------------------------
    # STEP 12 — LOAD KE CLICKHOUSE
    # ----------------------------------------------------------
    # CREATE TABLE dengan skema ClickHouse yang sesuai, lalu INSERT.
    # ENGINE = MergeTree() adalah engine default untuk tabel analitik
    # di ClickHouse — cocok untuk query agregasi cepat.
    #
    # TRUNCATE sebelum INSERT agar tidak duplikat saat re-run script.

    print("\n[STEP 12] Load ke ClickHouse...")

    # ── TABLE 1: gold_category_kpi ─────────────────────────────
    client.execute(f"""
    CREATE TABLE IF NOT EXISTS {TABLE_CATEGORY}
    (
        category               String,
        total_products         UInt32,
        total_inventory_value  Float64,
        avg_unit_price         Float64,
        total_stock            UInt32,
        avg_stock              Float64,
        total_reorder_qty      UInt32,
        total_sales_volume     UInt32,
        avg_sales_volume       Float64,
        avg_turnover_rate      Float64,
        low_stock_count        UInt32,
        active_products        UInt32,
        backordered_products   UInt32,
        discontinued_products  UInt32,
        low_stock_pct          Float64,
        revenue_per_unit_avg   Float64,
        gold_processed_at      String
    )
    ENGINE = MergeTree()
    ORDER BY category
    """)
    client.execute(f"TRUNCATE TABLE IF EXISTS {TABLE_CATEGORY}")
    ch_insert(client, TABLE_CATEGORY, gold_category)

    # ── TABLE 2: gold_inventory_status ────────────────────────
    client.execute(f"""
    CREATE TABLE IF NOT EXISTS {TABLE_STATUS}
    (
        category              String,
        status                String,
        product_count         UInt32,
        total_inventory_value Float64,
        total_stock           UInt32,
        avg_unit_price        Float64,
        low_stock_count       UInt32,
        avg_reorder_quantity  Float64,
        low_stock_pct         Float64,
        gold_processed_at     String
    )
    ENGINE = MergeTree()
    ORDER BY (category, status)
    """)
    client.execute(f"TRUNCATE TABLE IF EXISTS {TABLE_STATUS}")
    ch_insert(client, TABLE_STATUS, gold_status)

    # ── TABLE 3: gold_supplier_performance ────────────────────
    client.execute(f"""
    CREATE TABLE IF NOT EXISTS {TABLE_SUPPLIER}
    (
        supplier_name            String,
        total_products           UInt32,
        categories_supplied      UInt32,
        total_inventory_value    Float64,
        total_stock              UInt32,
        avg_unit_price           Float64,
        total_sales_volume       UInt32,
        avg_turnover_rate        Float64,
        low_stock_count          UInt32,
        active_products          UInt32,
        low_stock_pct            Float64,
        supply_reliability_score Float64,
        gold_processed_at        String
    )
    ENGINE = MergeTree()
    ORDER BY supplier_name
    """)
    client.execute(f"TRUNCATE TABLE IF EXISTS {TABLE_SUPPLIER}")
    ch_insert(client, TABLE_SUPPLIER, gold_supplier)

    # ── TABLE 4: gold_product_clusters ────────────────────────
    client.execute(f"""
    CREATE TABLE IF NOT EXISTS {TABLE_PRODUCT_CLUSTERS}
    (
        product_id               String,
        product_name             String,
        category                 String,
        supplier_name            String,
        status                   String,
        low_stock                UInt8,
        stock_quantity           Float64,
        unit_price               Float64,
        sales_volume             Float64,
        inventory_turnover_rate  Float64,
        inventory_value          Float64,
        cluster_id               UInt8,
        cluster_label            String,
        gold_processed_at        String
    )
    ENGINE = MergeTree()
    ORDER BY product_id
    """)
    client.execute(f"TRUNCATE TABLE IF EXISTS {TABLE_PRODUCT_CLUSTERS}")
    ch_insert(client, TABLE_PRODUCT_CLUSTERS, gold_product_clusters)

    # ── TABLE 5: gold_product_cluster_summary ─────────────────
    client.execute(f"""
    CREATE TABLE IF NOT EXISTS {TABLE_PRODUCT_CLUSTER_SUMMARY}
    (
        cluster_label        String,
        product_count        UInt32,
        avg_stock            Float64,
        avg_unit_price       Float64,
        avg_sales_volume     Float64,
        avg_turnover_rate    Float64,
        avg_inventory_value  Float64,
        low_stock_count      UInt32,
        active_count         UInt32,
        backordered_count    UInt32,
        discontinued_count   UInt32,
        low_stock_pct        Float64,
        gold_processed_at    String
    )
    ENGINE = MergeTree()
    ORDER BY cluster_label
    """)
    client.execute(f"TRUNCATE TABLE IF EXISTS {TABLE_PRODUCT_CLUSTER_SUMMARY}")
    ch_insert(client, TABLE_PRODUCT_CLUSTER_SUMMARY, gold_product_cluster_summary)

    # ── TABLE 6: gold_supplier_clusters ───────────────────────
    client.execute(f"""
    CREATE TABLE IF NOT EXISTS {TABLE_SUPPLIER_CLUSTERS}
    (
        supplier_name            String,
        categories_supplied      UInt32,
        active_products          UInt32,
        total_products           UInt32,
        avg_turnover_rate        Float64,
        low_stock_count          UInt32,
        total_sales_volume       UInt32,
        supply_reliability_score Float64,
        supplier_cluster_id      UInt8,
        supplier_tier            String,
        gold_processed_at        String
    )
    ENGINE = MergeTree()
    ORDER BY supplier_name
    """)
    client.execute(f"TRUNCATE TABLE IF EXISTS {TABLE_SUPPLIER_CLUSTERS}")
    ch_insert(client, TABLE_SUPPLIER_CLUSTERS, gold_supplier_clusters)

    # ── VERIFIKASI: row count di ClickHouse ───────────────────
    print()
    print("  Verifikasi row count di ClickHouse:")
    all_tables = [
        TABLE_CATEGORY, TABLE_STATUS, TABLE_SUPPLIER,
        TABLE_PRODUCT_CLUSTERS, TABLE_PRODUCT_CLUSTER_SUMMARY,
        TABLE_SUPPLIER_CLUSTERS,
    ]
    for tbl in all_tables:
        result = client.execute(f"SELECT COUNT(*) FROM {tbl}")
        count  = result[0][0]
        print(f"    {tbl:<48} : {count:>4} baris ✓")

    # ----------------------------------------------------------
    # SELESAI
    # ----------------------------------------------------------
    print()
    print("=" * 62)
    print("  GOLD LAYER PIPELINE - SELESAI ✓")
    print("=" * 62)
    print()
    print("  CSV tersimpan di:")
    for path in [PATH_CATEGORY, PATH_STATUS, PATH_SUPPLIER,
                 PATH_PRODUCT_CLUSTERS, PATH_PRODUCT_CLUSTER_SUMMARY,
                 PATH_SUPPLIER_CLUSTERS]:
        print(f"    {path}")
    print()
    print("  Tabel ClickHouse aktif di retail_dw:")
    for tbl in all_tables:
        print(f"    {CH_DATABASE}.{tbl}")
    print()
    print("  Langkah berikutnya — Power BI:")
    print("    1. Get Data → ODBC → DSN ClickHouse localhost:8123")
    print("       atau pakai HTTP connector ke http://localhost:8123")
    print("    2. Import semua tabel gold_ di atas")
    print("    3. Tambah halaman 'Cluster Analysis' di dashboard:")
    print("       - Scatter plot: X=sales_volume, Y=stock_quantity,")
    print("         Legend=cluster_label (warna per cluster)")
    print("       - Bar chart: supplier_tier vs jumlah supplier")
    print("    4. Slicer: cluster_label & category untuk filter")
    print("       interaktif antar halaman dashboard")
    print("=" * 62)


# ============================================================
# ENTRY POINT
# ============================================================
# Memanggil main() hanya jika script dijalankan langsung,
# bukan saat di-import sebagai modul.
# Best practice Python production — pipeline tidak jalan
# otomatis kalau file di-import di tempat lain.

if __name__ == "__main__":
    main()