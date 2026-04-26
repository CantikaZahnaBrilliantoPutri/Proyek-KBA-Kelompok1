# Proyek-KBA-F-Kelompok-1
## Infrastruktur
Proyek ini dikembangkan menggunakan tools berikut:
- **Editor** ---> Visual Studio Code (VS Code).
- **Bahasa** ---> Python 3.14.
- **Library Utama** ---> Pandas, SQLAlchemy, Openpyxl.
- **Sistem Operasi** ---> Windows 11.
## Penyiapan Project - Ingestion (Bronze)
1. Clone repositori
  ```
  git clone https://github.com/CantikaZahnaBrilliantoPutri/Proyek-KBA-F-Kelompok-1.git
  ```
2. Persiapan Virtual Environment
  Buka folder proyek di VS Code, buka terminal (Ctrl+`), lalu jalankan:
  ```
  python -m venv venv
  venv\Scripts\activate
  ```
3. Instalasi Library
   Gunakan versi pandas terbaru agar kompatibel dengan Python 3.14
  ```
  pip install pandas==2.3.3 sqlalchemy openpyxl
  ```
4. Menjalankan Bronze Pipeline
   Eksekusi script python untuk memindahkan data mentah ke folder output bronze
  ```
  python bronze/bronze_layer.py
  ```
5. Output Bronze Layer
   Data hasil ingestion akan muncul di folder `bronze_output/`

## Data Cleaning & Pre-Processing (Silver)
### Deskripsi Umum Silver Layer
Proses ini mengubah data dari Bronze Layer menjadi data yang bersih dan tervalidasi menggunakan Jupyter Notebook di VS Code.

#### Input (Bronze)
Mengambil file dari `bronze_transactions.csv`
1. **Load Data** Membaca dataset bronze ke dalam DataFrame.
2. **Data Cleaning**
    - Pengecekan missing values dan data duplikat.
    - Penghapusan baris yang tidak konsisten.
3. **Data Transformation**
    - Konversi tipe data kolom tanggal dan numerik.
    - Normalisasi teks pada kolom kategori.
4. **Feature Engineering**
    - Perhitungan `Inventory_Value` (Stock * Price).
    - Penentuan status `Low_Stock` secara otomatis.
5. **Data Validation**
    - Memastikan tidak ada nilai negatif pada kolom kuantitas dan harga.
    
6. **Pembuatan ID transaksi jika tidak tersedia**
    - Pada dataset transaksi, jika kolom `transaction_id` tidak ada, maka dibuat otomatis menggunakan `uuid()`

#### Output (Silver)
Hasil akhir disimpan dalam format CSV siap pakai `silver_inventory_notebook.ipynb`

### Cara Menjalankan Silver Layer
1. Buka file `silver_inventory.csv` di VS Code.
2. Pastikan Kernel di pojok kanan atas VS Code sudah mengarah ke `venv` (Python 3.14).
3. Jalankan sel satu per satu atau klik Run All
   - Step 1: Import library & Load data.
   - Step 2-3: Cleaning & Konversi tipe data.
   - Step 4: Validasi nilai numerik.
   - Step 5: Export ke Silver Layer.
5. Periksa folder `silver_inventory_notebook.ipynb` untuk melihat hasilnya.
