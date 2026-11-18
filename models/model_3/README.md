# Study MonkeyPox Classification

Klasifikasi 6 kelas citra lesi kulit (`Chickenpox`, `Cowpox`, `Healthy`, `HFMD`, `Measles`, `Monkeypox`) pada dataset MSLD v2.0 menggunakan pre-processing **LAB-CLAHE** dan ekstraksi fitur **EVA-02** (`eva02_tiny_patch14_224`) dengan skema **5-Fold Cross Validation**.

## Cara Menjalankan
1. Jalankan training & testing sekaligus: `python main.py`
2. Jalankan khusus training: `python main.py train`
3. Jalankan khusus testing: `python main.py test`