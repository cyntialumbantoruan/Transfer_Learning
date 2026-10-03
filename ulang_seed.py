"""Ulangi percobaan transfer vs scratch dengan beberapa seed, lalu rangkum.

Tiap seed mengubah DUA hal sekaligus: pembagian blok train/val/test
(siapkan_data.py) dan inisialisasi + urutan data latih (latih.py).

Pakai:
    python ulang_seed.py                       (seed 1-5, 11 kelas, dua mode)
    python ulang_seed.py --seeds 1 2 3 --kelas 4
    python ulang_seed.py --mode transfer

Aman dihentikan dan dijalankan lagi: percobaan yang sudah punya hasil.json
dilewati. Ringkasan disimpan di ringkasan_seed_k<kelas>.csv
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def jalankan(perintah):
    print("\n>>", " ".join(perintah), flush=True)
    hasil = subprocess.run(perintah)
    if hasil.returncode != 0:
        raise SystemExit(f"Perintah gagal (kode {hasil.returncode}), berhenti.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--kelas", type=int, choices=[11, 4], default=11)
    ap.add_argument("--mode", choices=["both", "transfer", "scratch"], default="both")
    ap.add_argument("--blok", type=int, default=5)
    args = ap.parse_args()

    modes = ["transfer", "scratch"] if args.mode == "both" else [args.mode]
    py = sys.executable
    hasil = {m: [] for m in modes}

    for seed in args.seeds:
        split = f"splits/split_seed{seed}.csv"
        jalankan([py, "siapkan_data.py", "--seed", str(seed),
                  "--blok", str(args.blok), "--keluaran", split])
        for mode in modes:
            folder = Path("hasil") / f"{mode}_k{args.kelas}_seed{seed}"
            berkas = folder / "hasil.json"
            if berkas.exists():
                print(f"\n(lewati {folder}, hasil sudah ada)")
            else:
                jalankan([py, "latih.py", "--mode", mode, "--kelas", str(args.kelas),
                          "--seed", str(seed), "--split", split])
            with open(berkas, encoding="utf-8") as fh:
                data = json.load(fh)
            hasil[mode].append((seed, data["test_acc"], data["test_macro_f1"]))

    # Ringkasan.
    print("\n================ RINGKASAN ================")
    print(f"kelas={args.kelas}, seed={args.seeds}\n")
    print(f"{'mode':10s} {'seed':>5s} {'akurasi':>9s} {'macro-F1':>9s}")
    baris_csv = []
    for mode in modes:
        for seed, acc, f1 in hasil[mode]:
            print(f"{mode:10s} {seed:5d} {acc:9.3f} {f1:9.3f}")
            baris_csv.append([mode, seed, f"{acc:.4f}", f"{f1:.4f}"])
    print()
    for mode in modes:
        acc = np.array([a for _, a, _ in hasil[mode]])
        f1 = np.array([f for _, _, f in hasil[mode]])
        sd = lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0
        print(f"{mode:10s} akurasi {acc.mean():.3f} ± {sd(acc):.3f} | "
              f"macro-F1 {f1.mean():.3f} ± {sd(f1):.3f}  (n={len(acc)} seed)")
        baris_csv.append([mode, "rata-rata", f"{acc.mean():.4f}", f"{f1.mean():.4f}"])
        baris_csv.append([mode, "std", f"{sd(acc):.4f}", f"{sd(f1):.4f}"])

    keluar = Path(f"ringkasan_seed_k{args.kelas}.csv")
    with open(keluar, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["mode", "seed", "test_akurasi", "test_macro_f1"])
        w.writerows(baris_csv)
    print(f"\nRingkasan tersimpan: {keluar.resolve()}")


if __name__ == "__main__":
    main()
