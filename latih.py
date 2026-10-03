"""Klasifikasi kondisi motor dari citra termal: transfer learning vs dari nol.

Dua mode (arsitektur sama: MobileNetV2, supaya perbandingannya adil):
  transfer : bobot pretrained ImageNet. Tahap 1 melatih lapisan klasifikasi saja,
             tahap 2 fine-tuning 4 blok terakhir dengan learning rate kecil.
  scratch  : bobot acak, semua lapisan dilatih.

Pakai:
    python latih.py --mode transfer --kelas 11
    python latih.py --mode scratch  --kelas 11
    python latih.py --mode transfer --kelas 4
Uji cepat (1 epoch):
    python latih.py --mode transfer --kelas 4 --epoch-head 1 --epoch-ft 1

Membaca split.csv (dibuat oleh siapkan_data.py). Hasil disimpan di folder hasil/.
Catatan: TIDAK memakai augmentasi warna/kecerahan, karena warna pada citra
termal membawa informasi suhu.
"""
import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

MEAN = [0.485, 0.456, 0.406]  # statistik ImageNet, dipakai model pretrained
STD = [0.229, 0.224, 0.225]


def baca_split(path_csv: str, kelas: int):
    """Baca split.csv -> (nama kelas, {split: [(path, label), ...]})."""
    kolom = "kelas11" if kelas == 11 else "kelas4"
    with open(path_csv, encoding="utf-8") as fh:
        baris = list(csv.DictReader(fh))
    nama_kelas = sorted({r[kolom] for r in baris})
    indeks = {k: i for i, k in enumerate(nama_kelas)}
    data = {"train": [], "val": [], "test": []}
    for r in baris:
        data[r["split"]].append((r["path"], indeks[r[kolom]]))
    return nama_kelas, data


class CitraTermal(Dataset):
    """Memuat semua gambar ke memori (datasetnya kecil)."""

    def __init__(self, daftar, transform):
        self.gambar, self.label = [], []
        for path, y in daftar:
            with Image.open(path) as im:
                self.gambar.append(im.convert("RGB"))
            self.label.append(y)
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i):
        return self.transform(self.gambar[i]), self.label[i]


def buat_model(mode: str, n_kelas: int):
    if mode == "transfer":
        bobot = models.MobileNet_V2_Weights.IMAGENET1K_V1  # diunduh otomatis (butuh internet)
        model = models.mobilenet_v2(weights=bobot)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_kelas)
    else:
        model = models.mobilenet_v2(weights=None, num_classes=n_kelas)
    return model


def evaluasi(model, loader, device, kriteria, n_kelas):
    model.eval()
    total, y_benar, y_duga = 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            keluar = model(x)
            total += kriteria(keluar, y).item() * len(y)
            y_duga += keluar.argmax(1).cpu().tolist()
            y_benar += y.cpu().tolist()
    akurasi = float(np.mean(np.array(y_benar) == np.array(y_duga)))
    f1 = float(f1_score(y_benar, y_duga, average="macro",
                        labels=list(range(n_kelas)), zero_division=0))
    return total / len(y_benar), akurasi, f1, y_benar, y_duga


def gambar_confusion(cm, nama_kelas, judul, path):
    fig, ax = plt.subplots(figsize=(0.7 * len(nama_kelas) + 3, 0.7 * len(nama_kelas) + 2.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(nama_kelas)))
    ax.set_yticks(range(len(nama_kelas)))
    ax.set_xticklabels(nama_kelas, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(nama_kelas, fontsize=8)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Sebenarnya")
    ax.set_title(judul, fontsize=10)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)


def gambar_kurva(riwayat, batas_tahap, path):
    ep = [r["epoch"] for r in riwayat]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(ep, [r["train_loss"] for r in riwayat], label="train")
    ax[0].plot(ep, [r["val_loss"] for r in riwayat], label="val")
    ax[0].set_title("Loss")
    ax[1].plot(ep, [r["train_acc"] for r in riwayat], label="train")
    ax[1].plot(ep, [r["val_acc"] for r in riwayat], label="val")
    ax[1].set_title("Akurasi")
    for a in ax:
        a.set_xlabel("epoch")
        a.legend()
        if batas_tahap:
            a.axvline(batas_tahap + 0.5, color="gray", linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["transfer", "scratch"], default="transfer")
    ap.add_argument("--kelas", type=int, choices=[11, 4], default=11)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epoch-head", type=int, default=8, help="transfer: tahap 1")
    ap.add_argument("--epoch-ft", type=int, default=12, help="transfer: tahap 2 (fine-tuning)")
    ap.add_argument("--epoch", type=int, default=20, help="scratch: jumlah epoch")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ukuran", type=int, default=224)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-ft", type=float, default=1e-4)
    ap.add_argument("--lr-scratch", type=float, default=1e-3)
    ap.add_argument("--split", default="split.csv")
    ap.add_argument("--keluaran", default="hasil")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nama_kelas, data = baca_split(args.split, args.kelas)
    n_kelas = len(nama_kelas)
    print(f"mode={args.mode} kelas={n_kelas} seed={args.seed} device={device}")
    print({k: len(v) for k, v in data.items()}, "->", nama_kelas)

    # Augmentasi hanya geometri ringan. Tanpa ColorJitter/hue (warna = suhu).
    t_train = transforms.Compose([
        transforms.Resize((args.ukuran, args.ukuran)),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    t_eval = transforms.Compose([
        transforms.Resize((args.ukuran, args.ukuran)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    g = torch.Generator().manual_seed(args.seed)
    loader_train = DataLoader(CitraTermal(data["train"], t_train), batch_size=args.batch,
                              shuffle=True, generator=g, num_workers=0)
    loader_val = DataLoader(CitraTermal(data["val"], t_eval), batch_size=args.batch, num_workers=0)
    loader_test = DataLoader(CitraTermal(data["test"], t_eval), batch_size=args.batch, num_workers=0)

    # Bobot kelas: kelas yang jarang diberi bobot lebih besar (penting untuk 4 kelas).
    cacah = np.bincount([y for _, y in data["train"]], minlength=n_kelas)
    bobot_kelas = torch.tensor(cacah.sum() / (n_kelas * np.maximum(cacah, 1)),
                               dtype=torch.float32).to(device)
    kriteria = nn.CrossEntropyLoss(weight=bobot_kelas)

    model = buat_model(args.mode, n_kelas).to(device)
    riwayat = []
    terbaik = {"f1": -1.0, "acc": -1.0, "epoch": 0, "state": None}

    def jalankan_tahap(nama, epoch, lr, parameter, atur_mode):
        optimizer = torch.optim.AdamW(parameter, lr=lr, weight_decay=1e-4)
        for _ in range(epoch):
            t0 = time.time()
            atur_mode()
            total, benar, n = 0.0, 0, 0
            for x, y in loader_train:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                keluar = model(x)
                loss = kriteria(keluar, y)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(y)
                benar += (keluar.argmax(1) == y).sum().item()
                n += len(y)
            v_loss, v_acc, v_f1, _, _ = evaluasi(model, loader_val, device, kriteria, n_kelas)
            nomor = len(riwayat) + 1
            riwayat.append({"epoch": nomor, "tahap": nama, "train_loss": total / n,
                            "train_acc": benar / n, "val_loss": v_loss,
                            "val_acc": v_acc, "val_f1": v_f1})
            if (v_f1, v_acc) > (terbaik["f1"], terbaik["acc"]):
                terbaik.update(f1=v_f1, acc=v_acc, epoch=nomor,
                               state=copy.deepcopy(model.state_dict()))
            print(f"[{nama}] epoch {nomor:2d} | train loss {total / n:.3f} acc {benar / n:.3f} "
                  f"| val loss {v_loss:.3f} acc {v_acc:.3f} f1 {v_f1:.3f} | {time.time() - t0:.0f} dtk")

    batas_tahap = 0
    if args.mode == "transfer":
        # Tahap 1: bekukan semua lapisan fitur, latih lapisan klasifikasi saja.
        for p in model.features.parameters():
            p.requires_grad = False

        def mode_tahap1():
            model.train()
            model.features.eval()  # statistik BatchNorm pretrained tetap

        jalankan_tahap("head", args.epoch_head, args.lr_head,
                       model.classifier.parameters(), mode_tahap1)
        batas_tahap = len(riwayat)

        # Tahap 2: buka 4 blok terakhir, latih dengan learning rate kecil.
        for p in model.features[-4:].parameters():
            p.requires_grad = True

        def mode_tahap2():
            model.train()
            for blok in model.features[:-4]:
                blok.eval()

        parameter = [p for p in model.parameters() if p.requires_grad]
        jalankan_tahap("finetune", args.epoch_ft, args.lr_ft, parameter, mode_tahap2)
    else:
        jalankan_tahap("scratch", args.epoch, args.lr_scratch,
                       model.parameters(), lambda: model.train())

    # Evaluasi test SEKALI, memakai epoch dengan F1 validasi terbaik.
    model.load_state_dict(terbaik["state"])
    _, t_acc, t_f1, y_benar, y_duga = evaluasi(model, loader_test, device, kriteria, n_kelas)
    cm = confusion_matrix(y_benar, y_duga, labels=list(range(n_kelas)))
    laporan = classification_report(y_benar, y_duga, labels=list(range(n_kelas)),
                                    target_names=nama_kelas, zero_division=0, output_dict=True)
    print(f"\nEpoch terbaik (val F1): {terbaik['epoch']}")
    print(f"TEST  akurasi {t_acc:.3f} | macro-F1 {t_f1:.3f}  (n test = {len(y_benar)})")
    print(classification_report(y_benar, y_duga, labels=list(range(n_kelas)),
                                target_names=nama_kelas, zero_division=0))

    folder = Path(args.keluaran) / f"{args.mode}_k{n_kelas}_seed{args.seed}"
    folder.mkdir(parents=True, exist_ok=True)
    gambar_confusion(cm, nama_kelas, f"{args.mode}, {n_kelas} kelas (test)", folder / "confusion_matrix.png")
    gambar_kurva(riwayat, batas_tahap, folder / "kurva.png")
    torch.save({"state_dict": terbaik["state"], "kelas": nama_kelas}, folder / "model.pt")
    with open(folder / "hasil.json", "w", encoding="utf-8") as fh:
        json.dump({"args": vars(args), "kelas": nama_kelas,
                   "jumlah": {k: len(v) for k, v in data.items()},
                   "epoch_terbaik": terbaik["epoch"], "val_f1_terbaik": terbaik["f1"],
                   "test_acc": t_acc, "test_macro_f1": t_f1,
                   "laporan": laporan, "confusion_matrix": cm.tolist(),
                   "riwayat": riwayat}, fh, indent=2)
    print(f"Hasil tersimpan di: {folder.resolve()}")


if __name__ == "__main__":
    main()
