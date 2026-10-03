"""Lihat struktur dataset gambar: folder, jumlah file, contoh nama, ukuran gambar.

Pakai:
    python cek_dataset.py            (membaca folder data)
    python cek_dataset.py data\\nama_folder_lain
"""
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

EKSTENSI_GAMBAR = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    if not root.exists():
        raise SystemExit(f"Folder tidak ditemukan: {root.resolve()}")

    print(f"Folder dataset: {root.resolve()}\n")
    total = 0
    ukuran = Counter()
    ekstensi_lain = Counter()

    # Telusuri semua folder, termasuk subfolder.
    for folder in sorted([root] + [p for p in root.rglob("*") if p.is_dir()]):
        gambar = sorted(
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in EKSTENSI_GAMBAR
        )
        for f in folder.iterdir():
            if f.is_file() and f.suffix.lower() not in EKSTENSI_GAMBAR:
                ekstensi_lain[f.suffix.lower() or "(tanpa ekstensi)"] += 1
        if not gambar:
            continue
        total += len(gambar)
        for f in gambar[:5]:  # ukuran dari beberapa contoh saja
            try:
                with Image.open(f) as im:
                    ukuran[(im.size, im.mode)] += 1
            except Exception as error:
                print(f"  [tidak bisa dibuka] {f.name}: {error}")
        nama = folder.relative_to(root) if folder != root else Path(".")
        print(f"{str(nama):45s} {len(gambar):5d} gambar   contoh: {gambar[0].name}")

    print(f"\nTotal gambar: {total}")
    if ukuran:
        print("Ukuran (lebar, tinggi) dan mode dari contoh:")
        for (size, mode), jumlah in ukuran.most_common():
            print(f"  {size} {mode}: {jumlah} contoh")
    if ekstensi_lain:
        print("File non-gambar:", dict(ekstensi_lain))


if __name__ == "__main__":
    main()
