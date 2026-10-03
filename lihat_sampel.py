"""Cek penomoran file dan buat gambar contoh tiap kelas.

Hasil:
  - tabel di terminal: jumlah, nomor terkecil/terbesar, dan loncatan nomor per kelas
  - sampel_kelas.png : 3 contoh (awal, tengah, akhir) dari tiap kelas

Pakai:  python lihat_sampel.py
"""
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # simpan ke file, tanpa membuka jendela
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data") / "IR-Motor-bmp"
KELUARAN = Path("sampel_kelas.png")


def nomor_dari_nama(nama: str):
    """Ambil angka dari nama file, misalnya 'f068.bmp' -> 68."""
    cocok = re.search(r"(\d+)", nama)
    return int(cocok.group(1)) if cocok else None


def main() -> None:
    if not ROOT.exists():
        raise SystemExit(f"Folder tidak ditemukan: {ROOT.resolve()}")

    kelas = sorted(p for p in ROOT.iterdir() if p.is_dir())
    print(f"{'kelas':12s} {'jumlah':>6s} {'min':>5s} {'maks':>5s} {'loncat':>7s}  contoh nama file")
    daftar = {}
    for folder in kelas:
        berkas = [f for f in folder.iterdir() if f.suffix.lower() == ".bmp"]
        pasangan = sorted((nomor_dari_nama(f.name), f) for f in berkas
                          if nomor_dari_nama(f.name) is not None)
        daftar[folder.name] = [f for _, f in pasangan]
        angka = [n for n, _ in pasangan]
        loncat = (max(angka) - min(angka) + 1 - len(angka)) if angka else 0
        contoh = ", ".join(f.name for _, f in pasangan[:4])
        print(f"{folder.name:12s} {len(angka):6d} {min(angka):5d} {max(angka):5d} {loncat:7d}  {contoh}, ...")

    # Gambar contoh: awal, tengah, akhir dari tiap kelas.
    baris, kolom = len(kelas), 3
    fig, sumbu = plt.subplots(baris, kolom, figsize=(7, 1.9 * baris))
    for i, folder in enumerate(kelas):
        berkas = daftar[folder.name]
        pilihan = [berkas[0], berkas[len(berkas) // 2], berkas[-1]]
        for j, f in enumerate(pilihan):
            ax = sumbu[i][j]
            with Image.open(f) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(folder.name, rotation=0, ha="right", va="center", fontsize=9)
            ax.set_title(f.name, fontsize=7)
    plt.tight_layout()
    plt.savefig(KELUARAN, dpi=110)
    print(f"\nGambar contoh tersimpan: {KELUARAN.resolve()}")


if __name__ == "__main__":
    main()
