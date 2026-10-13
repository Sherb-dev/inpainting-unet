import os
from multiprocessing import Pool

import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF
import cv2

from src.config import Config
from pathlib import Path
from src.dataset_builder import Dataset, DatasetBuilder   # al posto di MergedDatasetSplitter

SIZE, CACHE = Config.IMAGE_SIZE, Config.CACHE_DIR


def _save(img, path, interp):
    img = TF.resize(img, (SIZE, SIZE), interpolation=interp)
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

def _process(row):
    iid = row["image_id"]
    p = {k: os.path.join(CACHE, f"{iid}_{k}.png") for k in ("orig", "manip", "mask")}
    if all(os.path.exists(x) for x in p.values()):
        return 0                     # già presente

    B, N = TF.InterpolationMode.BILINEAR, TF.InterpolationMode.NEAREST
    _save(Image.open(row["orig_path"]).convert("RGB"), p["orig"], B)
    _save(Image.open(row["manip_path"]).convert("RGB"), p["manip"], B)

    # stessa conversione che fa oggi InpaintingDataset
    if os.path.splitext(row["mask_path"])[1] == ".npy":
        mask = Image.fromarray((np.load(_mask_border(row["mask_path"], row["dilatation_factor"])) * 255).astype(np.uint8)).convert("L")
    else:
        mask = Image.open(row["mask_path"]).convert("L")
    _save(mask, p["mask"], N)
    return 1

if __name__ == "__main__":
    cfg = Config()
    root = cfg.BASE_DIR / "dataset"
 
    datasets = []
    
    manifest = DatasetBuilder(datasets, cfg).run() # crea manifest_split.csv
    os.makedirs(CACHE, exist_ok=True)
    rows = manifest.to_dict("records")
    with Pool(max(1, os.cpu_count() // 2)) as pool:
        done = list(tqdm(pool.imap_unordered(_process, rows, chunksize=32), total=len(rows)))
    print(f"Cache pronta: {sum(done)} nuove coppie, {len(rows) - sum(done)} già presenti.")