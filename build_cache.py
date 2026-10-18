import os
from multiprocessing import Pool

import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF
import cv2
from functools import partial

from src.config import Config
from pathlib import Path
from src.dataset_builder import Dataset, DatasetBuilder   # al posto di MergedDatasetSplitter

# SIZE, CACHE = Config.IMAGE_SIZE, Config.CACHE_DIR


def _save(img, path, interp, cfg: Config):
    img = TF.resize(img, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), interpolation=interp)
    tmp = path + ".tmp"
    img.save(tmp, format="PNG")      # PNG lossless, mai JPEG
    os.replace(tmp, path)            # scrittura atomica

# Stessa identica funzione implementata durante l'inpainting 
def _mask_border(maschera_binaria: np.array, dilatation_factor: float):
    # calcolo il margine come il 1% della dimensione maggiore dell'immagine
    altezza, larghezza = maschera_binaria.shape
    dimensione_riferimento = max(altezza, larghezza)
    margine = int(dimensione_riferimento * dilatation_factor) # in teoria 0.01 per bagel e 0.03 per il resto

    # Importante -> deve essere dispari e almeno 3x3
    kernel_size = max(3, (margine * 2) + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    new_maschera_binaria = cv2.dilate(maschera_binaria, kernel, iterations=1)
    return new_maschera_binaria

def _process(row, cfg: Config):
    iid = row["image_id"]
    p = {k: os.path.join(cfg.CACHE_DIR, f"{iid}_{k}.png") for k in ("orig", "manip", "mask")}
    if all(os.path.exists(x) for x in p.values()):
        return 0                     # già presente

    B, N = TF.InterpolationMode.BILINEAR, TF.InterpolationMode.NEAREST
    _save(Image.open(row["orig_path"]).convert("RGB"), p["orig"], B, cfg)
    _save(Image.open(row["manip_path"]).convert("RGB"), p["manip"], B, cfg)

    # stessa conversione che fa oggi InpaintingDataset
    if os.path.splitext(row["mask_path"])[1] == ".npy":
        mask_dil = _mask_border(np.load(row["mask_path"]), row.get("dilatation_factor", 0.03))
        mask = Image.fromarray((mask_dil * 255).astype(np.uint8)).convert("L")
    else:
        mask = Image.open(row["mask_path"]).convert("L")
    _save(mask, p["mask"], N, cfg)
    return 1


def build_cache(cfg: Config, datasets:list[Dataset]): 
    _process_with_cfg = partial(_process, cfg=cfg)   
    
    manifest = DatasetBuilder(datasets, cfg).run() # crea manifest_split.csv
    os.makedirs(cfg.CACHE_DIR, exist_ok=True)
    rows = manifest.to_dict("records")
    with Pool(min(16, os.cpu_count() // 2)) as pool: # prima era max, ma comporta troppo overhead sull'I/O
        done = list(tqdm(pool.imap_unordered(_process_with_cfg, rows, chunksize=32), total=len(rows)))
    print(f"Cache pronta: {sum(done)} nuove coppie, {len(rows) - sum(done)} già presenti.")


if __name__ == "__main__":
    bagel = Dataset(
                    original_images_path="/workspace/Datasets/BAGEL",
                    manip_images_path="/workspace/Datasets/BAGEL_hdpainter",
                    mask_path_list="/workspace/Masks/BAGEL"
    )

    flux = Dataset(
                    original_images_path="/workspace/Datasets/flux1-krea",
                    manip_images_path="/workspace/Datasets/flux1-krea_hdpainter",
                    mask_path_list="/workspace/Masks/flux1-krea"
    )

    sense_nova = Dataset(
                    original_images_path="/workspace/Datasets/SenseNovaU1",
                    manip_images_path="/workspace/Datasets/SenseNovaU1_hdpainter",
                    mask_path_list="/workspace/Masks/SenseNovaU1"
    )

    BASE_DIR = Config.BASE_DIR
    IMAGE_SIZE = Config.IMAGE_SIZE

    cfg_bagel = Config()
    cfg_bagel.CACHE_DIR = f"{BASE_DIR}/dataset/cache_bagel_{IMAGE_SIZE}"
    cfg_bagel.MANIFEST_OUT = f"{BASE_DIR}/dataset/manifest_split_bagel.csv"

    cfg_flux = Config()
    cfg_flux.CACHE_DIR = f"{BASE_DIR}/dataset/cache_flux_{IMAGE_SIZE}"
    cfg_flux.MANIFEST_OUT = f"{BASE_DIR}/dataset/manifest_split_flux.csv"

    cfg_sensenova = Config()
    cfg_sensenova.CACHE_DIR = f"{BASE_DIR}/dataset/cache_sensenova_{IMAGE_SIZE}"
    cfg_sensenova.MANIFEST_OUT = f"{BASE_DIR}/dataset/manifest_split_sensenova.csv"

    build_cache(cfg_bagel, [bagel])
    build_cache(cfg_flux, [flux])
    build_cache(cfg_sensenova, [sense_nova])