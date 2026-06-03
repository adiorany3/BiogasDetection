# Streamlit Biogas Sapi Perah Indonesia

Aplikasi ini adalah versi yang sudah disesuaikan untuk konteks Indonesia.

## Fokus

Aplikasi hanya digunakan untuk:

- sapi perah / dairy;
- prediksi produksi biogas;
- input populasi ternak sapi perah;
- parameter lokal Indonesia;
- analisis faktor yang mempengaruhi produksi biogas.

## Footer dan Branding

Aplikasi sudah menyembunyikan elemen bawaan Streamlit seperti:

- main menu;
- header;
- footer bawaan;
- toolbar/status widget.

Footer custom:

```text
Created by Galuh Adi Insani with Kaggle data
(https://www.kaggle.com/datasets/mehmetisik/livestock-anaerobic-digester-database/data)
```

## Struktur Folder

```text
streamlit_biogas_sapi_perah_indonesia/
├── app.py
├── requirements.txt
├── README.md
└── data/
    └── agstar-livestock-ad-database.xlsx
```

## Cara Menjalankan

```bash
cd streamlit_biogas_sapi_perah_indonesia
pip install -r requirements.txt
streamlit run app.py
```

## Fitur Indonesia

Aplikasi sudah menambahkan parameter lokal:

- provinsi Indonesia;
- tipe biodigester lokal:
  - kubah tetap / fixed dome;
  - plastik tubular / balon;
  - lagoon tertutup;
  - CSTR;
  - plug flow;
  - biodigester beton komunal;
- skala usaha:
  - peternakan rakyat;
  - kelompok ternak / koperasi;
  - peternakan komersial;
  - unit komunal desa;
  - integrasi koperasi susu;
- kondisi iklim/lokasi kandang:
  - dataran rendah hangat;
  - dataran sedang;
  - dataran tinggi/sejuk;
- kotoran segar kg/ekor/hari;
- tingkat kotoran terkumpul;
- potensi biogas m³/kg kotoran;
- efisiensi biodigester;
- faktor manajemen operasi.

## Metode Prediksi

Model memakai pendekatan hybrid:

```text
Prediksi final =
    baseline Kaggle/AGSTAR × bobot baseline
    + estimasi lokal Indonesia × bobot lokal
```

Estimasi lokal Indonesia:

```text
Biogas lokal =
    Populasi sapi perah
    × kotoran kg/ekor/hari
    × tingkat kotoran terkumpul
    × potensi biogas m³/kg
    × efisiensi biodigester
    × faktor iklim
    × faktor manajemen operasi
```

## Catatan

Data Kaggle/AGSTAR digunakan sebagai baseline internasional. Untuk hasil paling akurat,
parameter lokal sebaiknya dikalibrasi dengan data lapangan peternakan sapi perah di Indonesia.


## Perbaikan Desain Theme-Ready

Versi ini sudah diperbaiki agar mudah dibaca pada tema light dan dark/night:

- warna card mengikuti `--secondary-background-color`;
- warna teks mengikuti `--text-color`;
- warna link mengikuti `--primary-color`;
- metric, tabel, code block, sidebar, dan footer memiliki border kontras adaptif;
- footer custom tetap terbaca di light maupun dark mode;
- hero/header tidak lagi memakai warna latar terang permanen.


## Fix Keterbacaan Dark/Light

Versi ini memperbaiki masalah tulisan tidak terbaca pada dark mode dengan cara:

- menghapus override global warna teks;
- membiarkan komponen bawaan Streamlit mengikuti tema bawaan;
- membuat elemen custom seperti hero dan footer memakai warna eksplisit high-contrast;
- memastikan teks pada hero, card, dan footer selalu terang di atas background gelap;
- mempertahankan border lembut pada metric dan dataframe tanpa memaksa warna teks.
