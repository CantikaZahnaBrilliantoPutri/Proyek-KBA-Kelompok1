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
         + semua tabel di atas masuk ke ClickHouse (retail_dw)
 
CARA JALANKAN:
    pip install clickhouse-driver
    python gold_layer_steps.py
============================================================
"""
 
import pandas as pd
import numpy as np
import os
from datetime import datetime
from clickhouse_driver import Client
 
 
# ============================================================
# STEP 1 — KONFIGURASI
# ============================================================
# Sesuaikan path file dan setting ClickHouse dengan environment
# kalian sebelum menjalankan script ini.
 
SILVER_FILE = "silver_inventory.csv"   # letakkan di folder yang sama
OUTPUT_DIR  = "gold_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
# Setting ClickHouse — ganti sesuai environment kalian
CH_HOST     = "alamat-host-kamu-dari-cloud.clickhouse.cloud" # Ganti ini
CH_PORT     = 9440                                          # Port default Cloud
CH_USER     = "default"                                     # Biasanya 'default'
CH_PASSWORD = "password-rahasia-kamu"                        # Ganti ini
CH_DATABASE = "retail_dw"
CH_SECURE   = True                                          # Wajib True untuk Cloud

# Nama tabel Gold di ClickHouse
TABLE_CATEGORY = "gold_category_kpi"
TABLE_STATUS   = "gold_inventory_status"
TABLE_SUPPLIER = "gold_supplier_performance"
 
print("=" * 60)
print("  GOLD LAYER PIPELINE - START")
print("  Proyek: Analisis Tren Pembelian Pelanggan Retail")
print("  Kelompok 1 - KBA-F, Universitas Brawijaya 2026")
print("=" * 60)
print(f"  Silver input : {SILVER_FILE}")
print(f"  Output dir   : {OUTPUT_DIR}/")
print(f"  ClickHouse   : {CH_HOST}:{CH_PORT} / {CH_DATABASE}")
print("=" * 60)
 
 
# ============================================================
# STEP 2 — LOAD SILVER DATA
# ============================================================
# Baca silver_inventory.csv sebagai input Gold Layer.
# File ini adalah output bersih dari Silver Layer.
 
print("\n[STEP 2] Load Silver Data...")
 
if not os.path.exists(SILVER_FILE):
    raise FileNotFoundError(
        f"File '{SILVER_FILE}' tidak ditemukan.\n"
        "Pastikan silver_inventory.csv ada di folder yang sama."
    )
 
df = pd.read_csv(SILVER_FILE)
 
print(f"  ✓ Silver data dimuat: {df.shape[0]:,} baris × {df.shape[1]} kolom")
print(f"  Kolom: {list(df.columns)}")
 
 
# ============================================================
# STEP 3 — CLEANING & CAST TYPES
# ============================================================
# Ada beberapa kolom dari Silver yang perlu dibenerin tipenya
# sebelum bisa diagregasi di Gold Layer:
#
#   - 'percentage'   : masih String format '1.96%' → float
#   - kolom tanggal  : masih String → datetime
#   - 'Low_Stock'    : pastikan boolean
 
print("\n[STEP 3] Cleaning & Cast Types...")
 
GOLD_TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
 
# 3a. Bersihkan kolom percentage — hapus '%' lalu convert ke float
df["percentage"] = (
    df["percentage"]
    .str.replace("%", "", regex=False)
    .astype(float)
)
print("  ✓ percentage → float")
 
# 3b. Parse kolom tanggal ke datetime
date_cols = ["Date_Received", "Last_Order_Date", "Expiration_Date"]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce")
    nulls = df[col].isna().sum()
    print(f"  ✓ {col} → datetime | null: {nulls}")
 
# 3c. Pastikan Low_Stock bertipe boolean
df["Low_Stock"] = df["Low_Stock"].astype(bool)
print("  ✓ Low_Stock → bool")
 
print(f"  ✓ Timestamp Gold: {GOLD_TIMESTAMP}")
 
 
# ============================================================
# STEP 4 — GOLD TABLE 1: gold_category_kpi
# ============================================================
# Agregasi KPI utama per kategori produk.
# Tabel ini yang paling sering dikonsumsi Dashboard KPI.
#
# KPI yang dihasilkan:
#   total_products        — jumlah produk unik per kategori
#   total_inventory_value — total nilai inventori
#   avg_unit_price        — rata-rata harga satuan
#   total_stock           — total stok tersedia
#   total_sales_volume    — total volume penjualan
#   avg_turnover_rate     — rata-rata inventory turnover
#   low_stock_count       — jumlah produk stok rendah
#   low_stock_pct         — persentase produk low stock
#   active/backordered/discontinued_products — distribusi status
#   revenue_per_unit_avg  — estimasi revenue per unit
 
print("\n[STEP 4] Membuat gold_category_kpi...")
 
gold_category = (
    df.groupby("Catagory")
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
    .rename(columns={"Catagory": "category"})
)
 
# Derived metrics
gold_category["low_stock_pct"] = (
    gold_category["low_stock_count"] / gold_category["total_products"] * 100
).round(2)
 
gold_category["revenue_per_unit_avg"] = (
    gold_category["total_inventory_value"]
    / gold_category["total_stock"].replace(0, np.nan)
).round(2)
 
# Bulatkan kolom numerik
for col in ["avg_unit_price", "avg_stock", "avg_sales_volume", "avg_turnover_rate"]:
    gold_category[col] = gold_category[col].round(2)
 
# Tambah metadata
gold_category["gold_processed_at"] = GOLD_TIMESTAMP
 
print(f"  ✓ gold_category_kpi: {gold_category.shape[0]} baris × {gold_category.shape[1]} kolom")
print(gold_category[["category", "total_products", "total_inventory_value",
                      "total_sales_volume", "low_stock_pct"]].to_string(index=False))
 
 
# ============================================================
# STEP 5 — GOLD TABLE 2: gold_inventory_status
# ============================================================
# Agregasi per kombinasi Kategori × Status produk.
# Berguna untuk analisis operasional — misal berapa produk
# Backordered di Dairy, atau nilai inventori Discontinued.
 
print("\n[STEP 5] Membuat gold_inventory_status...")
 
gold_status = (
    df.groupby(["Catagory", "Status"])
    .agg(
        product_count         = ("Product_ID",      "count"),
        total_inventory_value = ("Inventory_Value", "sum"),
        total_stock           = ("Stock_Quantity",  "sum"),
        avg_unit_price        = ("Unit_Price",       "mean"),
        low_stock_count       = ("Low_Stock",        "sum"),
        avg_reorder_quantity  = ("Reorder_Quantity", "mean"),
    )
    .reset_index()
    .rename(columns={"Catagory": "category", "Status": "status"})
)
 
gold_status["low_stock_pct"] = (
    gold_status["low_stock_count"] / gold_status["product_count"] * 100
).round(2)
gold_status["avg_unit_price"]       = gold_status["avg_unit_price"].round(2)
gold_status["avg_reorder_quantity"] = gold_status["avg_reorder_quantity"].round(1)
gold_status["gold_processed_at"]    = GOLD_TIMESTAMP
 
print(f"  ✓ gold_inventory_status: {gold_status.shape[0]} baris × {gold_status.shape[1]} kolom")
print(gold_status[["category", "status", "product_count",
                    "total_inventory_value", "low_stock_pct"]].to_string(index=False))
 
 
# ============================================================
# STEP 6 — GOLD TABLE 3: gold_supplier_performance
# ============================================================
# Agregasi performa per supplier.
# Berguna untuk Tim Procurement — supplier mana yang paling
# reliable berdasarkan sales volume, turnover, dan low stock.
#
# supply_reliability_score (0–100) dihitung dari:
#   40% — total_sales_volume    (tinggi = bagus)
#   30% — avg_turnover_rate     (tinggi = bagus)
#   30% — low_stock_pct         (rendah = bagus, dibalik)
 
print("\n[STEP 6] Membuat gold_supplier_performance...")
 
gold_supplier = (
    df.groupby("Supplier_Name")
    .agg(
        total_products        = ("Product_ID",              "count"),
        categories_supplied   = ("Catagory",                "nunique"),
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
 
gold_supplier["low_stock_pct"]    = (gold_supplier["low_stock_count"] / gold_supplier["total_products"] * 100).round(2)
gold_supplier["avg_unit_price"]   = gold_supplier["avg_unit_price"].round(2)
gold_supplier["avg_turnover_rate"]= gold_supplier["avg_turnover_rate"].round(2)
 
# Normalisasi min-max untuk reliability score
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
 
print(f"  ✓ gold_supplier_performance: {gold_supplier.shape[0]} baris × {gold_supplier.shape[1]} kolom")
print("  Top 5 Supplier:")
print(gold_supplier[["supplier_name", "total_products",
                      "total_sales_volume", "supply_reliability_score"]]
      .head(5).to_string(index=False))
 
 
# ============================================================
# STEP 7 — SIMPAN KE CSV (LOCAL BACKUP)
# ============================================================
# Simpan dulu ke CSV lokal sebelum load ke ClickHouse.
# Ini penting sebagai backup — kalau koneksi ClickHouse gagal,
# data Gold tetap tersedia di folder gold_output/.
 
print("\n[STEP 7] Simpan ke CSV lokal...")
 
PATH_CATEGORY = os.path.join(OUTPUT_DIR, "gold_category_kpi.csv")
PATH_STATUS   = os.path.join(OUTPUT_DIR, "gold_inventory_status.csv")
PATH_SUPPLIER = os.path.join(OUTPUT_DIR, "gold_supplier_performance.csv")
 
gold_category.to_csv(PATH_CATEGORY, index=False)
gold_status.to_csv(PATH_STATUS,     index=False)
gold_supplier.to_csv(PATH_SUPPLIER, index=False)
 
print(f"  ✓ {PATH_CATEGORY}  — {len(gold_category)} baris")
print(f"  ✓ {PATH_STATUS}    — {len(gold_status)} baris")
print(f"  ✓ {PATH_SUPPLIER}  — {len(gold_supplier)} baris")
  
# ============================================================
# SELESAI
# ============================================================
print()
print("=" * 60)
print("  GOLD LAYER PIPELINE - SELESAI ✓")
print("=" * 60)
print()
print("  Output tersimpan di:")
print(f"    {PATH_CATEGORY}")
print(f"    {PATH_STATUS}")
print(f"    {PATH_SUPPLIER}")
print()
print("  Tabel di ClickHouse:")
print(f"    {CH_DATABASE}.{TABLE_CATEGORY}")
print(f"    {CH_DATABASE}.{TABLE_STATUS}")
print(f"    {CH_DATABASE}.{TABLE_SUPPLIER}")
print()
print("  Langkah berikutnya:")
print("    Koneksikan tabel Gold ke Power BI untuk Dashboard KPI.")
print("=" * 60)