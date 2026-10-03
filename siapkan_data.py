"""Bagi dataset menjadi train / validasi / test per BLOK gambar berurutan.

Kenapa per blok: gambar yang diambil berurutan sangat mirip. Kalau dibagi acak
per gambar, gambar yang hampir kembar bisa masuk train dan test sekaligus,
sehingga akurasi tampak terlalu bagus.

Pakai:
    python siapkan_data.py                    (seed 42, blok 5 gambar)
    python siapkan_data.py --seed 7 --blok 4
    python siapkan_data.py --seed 7 --keluaran splits/split_seed7.csv

Hasil: split.csv  (kolom: path, kelas11, kelas4, split, blok)
"""
import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path("data") / "IR-Motor-bmp"

# Penggabungan 11 kelas menjadi 4 kelas untuk laporan tambahan.
KELAS4 = {"Noload": "Sehat", "Fan": "Kipas", "Rotor-0": "Rotor_macet"}


def nomor(nama: str) -> int:
    return int(re.search(r"(\d+)", nama).group(1))


def bagi_blok(jumlah_blok: int, rng: random.Random) -> dict:
    """Tentukan blok mana masuk train/val/test (kira-kira 60/20/20)."""
    urutan = list(range(jumlah_blok))
    rng.shuffle(urutan)
    n_test = max(1, round(0.2 * jumlah_blok))
    n_val = max(1, round(0.2 * jumlah_blok))
    peta = {}
    for i, b in enumerate(urutan):
        peta[b] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return peta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--blok", type=int, default=5, help="gambar berurutan per blok")
    ap.add_argument("--keluaran", default="split.csv", help="nama file hasil (default: split.csv)")
    args = ap.parse_args()

    if not ROOT.exists():
        raise SystemExit(f"Folder tidak ditemukan: {ROOT.resolve()}")

    rng = random.Random(args.seed)
    baris = []
    hitung = defaultdict(lambda: defaultdict(int))

    for folder in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        berkas = sorted((f for f in folder.iterdir() if f.suffix.lower() == ".bmp"),
                        key=lambda f: nomor(f.name))
        # Potong jadi blok berurutan; sisa kecil digabung ke blok sebelumnya.
        blok = [berkas[i:i + args.blok] for i in range(0, len(berkas), args.blok)]
        if len(blok) > 1 and len(blok[-1]) < 3:
            blok[-2].extend(blok.pop())
        peta = bagi_blok(len(blok), rng)
        for b, isi in enumerate(blok):
            for f in isi:
                baris.append({
                    "path": f.as_posix(),
                    "kelas11": folder.name,
                    "kelas4": KELAS4.get(folder.name, "Hubung_singkat_stator"),
                    "split": peta[b],
                    "blok": f"{folder.name}_{b}",
                })
                hitung[folder.name][peta[b]] += 1

    keluaran = Path(args.keluaran)
    keluaran.parent.mkdir(parents=True, exist_ok=True)
    with open(keluaran, "w", newline="", encoding="utf-8") as fh:
        penulis = csv.DictWriter(fh, fieldnames=["path", "kelas11", "kelas4", "split", "blok"])
        penulis.writeheader()
        penulis.writerows(baris)

    print(f"seed={args.seed}, blok={args.blok}\n")
    print(f"{'kelas':12s} {'train':>6s} {'val':>5s} {'test':>5s} {'total':>6s}")
    total = defaultdict(int)
    for kelas in sorted(hitung):
        t, v, s = (hitung[kelas][k] for k in ("train", "val", "test"))
        print(f"{kelas:12s} {t:6d} {v:5d} {s:5d} {t + v + s:6d}")
        total["train"] += t
        total["val"] += v
        total["test"] += s
    print(f"{'JUMLAH':12s} {total['train']:6d} {total['val']:5d} {total['test']:5d} "
          f"{sum(total.values()):6d}")
    print(f"\nTersimpan: {keluaran.resolve()}")


if __name__ == "__main__":
    main()
