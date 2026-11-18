# Modified CNN — Mpox Skin Lesion Classification

Project ini merupakan pemisahan notebook `Untitled43.ipynb` menjadi beberapa file Python.

## Struktur folder

```text
Modified_CNN_Mpox_Project_Revisi/
├── data/
│   └── README.md
├── outputs/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── main.py
├── README.md
└── requirements.txt
```

## Menjalankan

Buka folder project ini di VS Code, buka terminal pada folder utama, lalu jalankan:

```bash
pip install -r requirements.txt
python -m src.main
```

Dataset akan diunduh otomatis melalui KaggleHub. Pastikan koneksi internet tersedia.

## Pembagian file

- `config.py`: hyperparameter, device, dan lokasi hasil.
- `dataset.py`: download dataset, melihat struktur folder, menentukan path, dan preprocessing.
- `model.py`: arsitektur Modified CNN dari notebook.
- `train.py`: loop training 5-fold, validation, testing, grafik, classification report, confusion matrix, dan penyimpanan metrik.
- `main.py`: menjalankan seluruh proses.

Hasil training disimpan ke folder `Hasil_5_Fold_Model2`.

**Catatan:** kode training dan evaluasi berasal dari cell notebook asli dan dipindahkan ke fungsi agar dapat dipanggil dari `main.py`. Hasil numerik baru muncul setelah program dijalankan.
