# Transfer Learning untuk Klasifikasi Kondisi Motor dari Citra Termal

Tugas Computer Vision and Deep Learning (RET503): membangun model dengan metode
transfer learning menggunakan data yang relevan dengan proyek PBL.

## Kaitan dengan proyek PBL

Proyek PBL kami adalah robot quadruped 12-DOF untuk inspeksi kualitas udara dan termal indoor. Salah satu kemampuan yang dibutuhkan robot adalah mengenali kondisi termal peralatan listrik, misalnya motor yang bekerja tidak normal, agar titik yang perlu diperiksa dapat ditandai. Pada semester ini target proyek adalah robot fisik dan perkabelan elektrikal, sehingga robot dan kamera termalnya belum dapat dipakai mengumpulkan data. Karena itu model dilatih dengan dataset termal publik motor induksi. Model ini disiapkan sebagai calon modul persepsi termal, dan perlu dilatih ulang dengan data dari kamera robot setelah kamera tersedia. 

## Dataset

Thermal image of equipment (Induction Motor), Babol Noshirvani University of Technology.

- Najafi, M., Baleghi, Y., Mirimani, S. M. (2021). *Thermal image of equipment (Induction Motor)*.
  Mendeley Data, V2. doi: [10.17632/m4sbt8hbvk.2](https://doi.org/10.17632/m4sbt8hbvk.2)
- Lisensi: Creative Commons Attribution 4.0 International (CC BY 4.0), (https://creativecommons.org/licenses/by/4.0/). Dataset tidak dimodifikasi; gambar hanya diubah ukurannya di dalam kode saat pelatihan.
- 369 gambar termal berwarna (BMP, 320x240 piksel pada contoh yang diperiksa), 11 kondisi motor
  induksi tiga fasa. Pemetaan nama folder ke kondisi dicocokkan dari jumlah gambar pada
  tabel di dokumen Readme dataset:

| Folder | Jumlah | Kondisi |
|---|---|---|
| A10, A30, A50 | 34, 37, 35 | Hubung singkat stator 1 fasa, tingkat 10%, 30%, 50% |
| A&C10, A&C30, A&B50 | 31, 38, 38 | Hubung singkat stator 2 fasa, tingkat 10%, 30%, 50% |
| A&C&B10, A&C&B30 | 31, 42 | Hubung singkat stator 3 fasa, tingkat 10%, 30% |
| Fan | 28 | Kegagalan pendingin |
| Rotor-0 | 30 | Rotor macet |
| Noload | 25 | Sehat |

Dataset **tidak disertakan** di repositori ini. Unduh dari tautan di atas, lalu letakkan
folder `IR-Motor-bmp` di dalam folder `data/`.

## Metode

- **Model**: MobileNetV2, input 224x224. Dua mode dengan arsitektur yang sama:
  - `transfer`: bobot pretrained ImageNet. Tahap 1 (8 epoch, lr 1e-3) melatih lapisan
    klasifikasi saja. Tahap 2 (12 epoch, lr 1e-4) fine-tuning 4 blok terakhir.
  - `scratch`: bobot acak, semua lapisan dilatih 20 epoch (lr 1e-3).
- Optimizer AdamW (weight decay 1e-4), batch 16, loss cross-entropy dengan bobot kelas.
- Augmentasi hanya geometri ringan (rotasi 5 derajat, geser 5%, skala 0,95-1,05).
  Tanpa perubahan warna atau kecerahan, karena warna pada citra termal membawa informasi suhu.
- **Pembagian data**: gambar yang diambil berurutan sangat mirip, sehingga pembagian acak
  per gambar akan bocor. Di tiap kelas, gambar diurutkan lalu dikelompokkan menjadi blok
  berisi 5 gambar berurutan, dan **blok** dibagi acak ke train/validasi/test (sekitar 60/20/20).
- Epoch terbaik dipilih dari macro-F1 validasi. Data test dipakai satu kali di akhir.
- Percobaan diulang dengan seed 1-5. Tiap seed mengubah pembagian blok sekaligus
  inisialisasi model. Percobaan awal memakai seed 42.

## Hasil (11 kelas, data test)

Rata-rata ± simpangan baku sampel dari 5 seed:

| Mode | Akurasi | Macro-F1 |
|---|---|---|
| Transfer learning | 0,958 ± 0,033 | 0,931 ± 0,056 |
| Dari nol (scratch) | 0,780 ± 0,100 | 0,695 ± 0,123 |

Per seed:

| Seed | Transfer: akurasi | Transfer: macro-F1 | Scratch: akurasi | Scratch: macro-F1 |
|---|---|---|---|---|
| 1 | 1,000 | 1,000 | 0,859 | 0,788 |
| 2 | 0,932 | 0,882 | 0,712 | 0,575 |
| 3 | 0,929 | 0,879 | 0,700 | 0,642 |
| 4 | 0,986 | 0,982 | 0,917 | 0,860 |
| 5 | 0,943 | 0,913 | 0,714 | 0,609 |

Percobaan awal (seed 42, `split.csv`): transfer 1,000 / 1,000, scratch 0,729 / 0,684
(akurasi / macro-F1).

Gambar confusion matrix dan kurva pelatihan ada di `hasil/<mode>_k11_seed<seed>/`.

### Pengamatan

- Transfer learning lebih tinggi daripada scratch pada semua 5 seed, dan simpangannya lebih kecil.
- Waktu pelatihan di CPU laptop: sekitar 3 menit per percobaan untuk transfer dan sekitar
  7-9 menit untuk scratch (dari log pelatihan).
- Pada scratch, akurasi validasi pada 4 epoch pertama hanya 0,07-0,09 di semua seed dan
  loss validasi sering melonjak. Penyebabnya tidak diselidiki.
- Kelas yang paling sering salah pada kedua mode: hubung singkat 10% (A10, A&C10) dan Noload.
  - Pada transfer learning, kesalahan di seed 2, 3, dan 5 hanya satu jenis: gambar A10 diprediksi sebagai A&C10 (5 dari 5 gambar pada seed 2 dan 3, 4 dari 5 pada seed 5). Seed 1 tanpa kesalahan, dan pada seed 4 hanya satu gambar Noload yang salah (diprediksi sebagai A30, disimpulkan dari laporan klasifikasi).
- Pada scratch (seed 42), seluruh 10 gambar A&B50 diprediksi sebagai A&C&B30, seluruh 5 gambar A30 sebagai A&C&B10, dan 4 dari 5 gambar Noload (sehat) sebagai A&C&B10. Kesalahan sehat dibaca sebagai hubung singkat juga muncul pada transfer seed 4.
- Dugaan kami, kelas-kelas ini tampak serupa pada citra termal berwarna, tetapi hal ini belum diuji.

## Keterbatasan

- Train, validasi, dan test berasal dari rekaman yang sama (satu sesi per kondisi). Pembagian
  per blok mengurangi kemiripan antar gambar, tetapi tidak menguji kemampuan pada rekaman
  atau motor baru. Hasil mendekati 1,0 tidak boleh dibaca sebagai akurasi di lapangan.
- Data test kecil (70-73 gambar, 5-10 gambar per kelas), jadi satu gambar yang salah
  mengubah recall kelas cukup besar.
- Lima seed memakai 369 gambar yang sama dengan pembagian blok berbeda, sehingga tidak
  dapat dianggap percobaan yang independen. Tidak dilakukan uji signifikansi.
- Dataset diambil di laboratorium dengan satu kamera termal; citra berupa pseudo-color,
  bukan nilai suhu mentah. Model belum diuji pada kamera atau ruangan robot.
- Pemetaan nama folder ke kondisi didasarkan pada kecocokan jumlah gambar dengan tabel
  dataset, bukan keterangan eksplisit penulis dataset.

## Struktur repositori

```
latih.py           pelatihan dan evaluasi (mode transfer / scratch)
siapkan_data.py    pembagian train/val/test per blok -> CSV
ulang_seed.py      mengulang percobaan untuk beberapa seed dan merangkum hasil
cek_dataset.py     melihat struktur dataset
lihat_sampel.py    cek penomoran file dan membuat gambar contoh per kelas
requirements.txt
splits/            file pembagian data tiap seed
hasil/             confusion matrix, kurva, hasil.json per percobaan
ringkasan_seed_k11.csv
```

## Cara menjalankan

Versi library yang dipakai tercatat di `requirements.txt` (Python 3.14, CPU).

```
pip install -r requirements.txt
# letakkan dataset di data/IR-Motor-bmp

python siapkan_data.py                       # membuat split.csv (seed 42)
python latih.py --mode transfer --kelas 11
python latih.py --mode scratch  --kelas 11

python ulang_seed.py                         # seed 1-5, kedua mode
```

Bobot pretrained MobileNetV2 diunduh otomatis oleh torchvision saat pertama kali dijalankan.
