# CNN Mpox Skin Lesion Classification — Project Gabungan

Project ini menggabungkan tiga implementasi model klasifikasi citra lesi kulit Mpox dalam satu folder. Kode masing-masing model tetap dipisahkan agar arsitektur, konfigurasi, dan proses training tidak saling tertukar.

## Daftar Model

1. **Model 1 — SimpleCNN**: kode utama berasal dari `ipsd_kel_VSCode.py` yang disediakan kelompok Model 1. File tersebut ditempatkan sebagai `models/model_1/main.py`. Folder `Hasil_5_Fold/` berisi hasil evaluasi yang sudah disertakan.
2. **Model 2 — ModifiedCNN**: project pemisahan notebook menjadi file Python (`config.py`, `dataset.py`, `model.py`, `train.py`, dan `main.py`).
3. **Model 3 — EVA-02**: project `Study MonkeyPox Classification`, menggunakan backbone EVA-02 dan classifier head, dengan preprocessing LAB-CLAHE.

## Struktur Folder

```text
CNN_Mpox_Combined_Project/
├── models/
│   ├── model_1/
│   │   ├── main.py
│   │   └── Hasil_5_Fold/
│   ├── model_2/
│   │   ├── src/
│   │   ├── data/
│   │   ├── outputs/
│   │   └── README.md
│   └── model_3/
│       ├── main.py
│       ├── src/
│       ├── assets/
│       └── outputs/
├── requirements.txt
└── README.md
```

## Instalasi

Disarankan menggunakan virtual environment. Dari folder utama project, jalankan:

```bash
python -m venv .venv
```

Aktifkan environment di Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install seluruh dependensi:

```bash
pip install -r requirements.txt
```

## Cara Menjalankan

Jalankan perintah dari folder utama project. Setiap model dijalankan dari foldernya sendiri agar import `src` pada Model 2 dan Model 3 tetap mengarah ke package yang benar.

### Model 1 — SimpleCNN

```powershell
cd models/model_1
python main.py
```

### Model 2 — ModifiedCNN

```powershell
cd models/model_2
python -m src.main
```

### Model 3 — EVA-02

```powershell
cd models/model_3
python main.py
```

Untuk kembali ke folder utama dari folder model, gunakan:

```powershell
cd ../..
```

## Dataset

Ketiga implementasi menggunakan dataset Mpox Skin Lesion Dataset melalui KaggleHub atau pencarian folder dataset sesuai kode masing-masing. Pastikan akses internet dan konfigurasi Kaggle yang diperlukan tersedia. Dataset tidak digabungkan ke dalam ZIP ini karena ukurannya besar.

## Hasil dan Catatan

- Hasil yang sudah tersedia dari ZIP asal tetap ditempatkan di folder model masing-masing.
- Output baru akan disimpan mengikuti konfigurasi/path yang sudah ditentukan pada masing-masing implementasi.
- Hyperparameter, preprocessing, arsitektur, dan alur training setiap model tidak diseragamkan oleh proses penggabungan ini.
- Perbandingan performa sebaiknya dilakukan setelah memastikan dataset, pembagian data, dan metrik evaluasi yang digunakan memang sebanding.

## Dependencies

Daftar package gabungan tersedia di `requirements.txt`. File tersebut merupakan gabungan dependensi yang tercantum pada project Model 2 dan Model 3 serta kebutuhan yang terlihat dari kode Model 1.
