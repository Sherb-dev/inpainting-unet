import os
from multiprocessing import Pool

import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF

from src.config import Config
from src.dataset_builder import MergedDatasetSplitter

SIZE, CACHE = Config.IMAGE_SIZE, Config.CACHE_DIR


def _save(img, path, interp):
    img = TF.resize(img, (SIZE, SIZE), interpolation=interp)
    tmp = path + ".tmp"
    img.save(tmp, format="PNG")      # PNG lossless, mai JPEG
    os.replace(tmp, path)            # scrittura atomica


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
        mask = Image.fromarray((np.load(row["mask_path"]) * 255).astype(np.uint8)).convert("L")
    else:
        mask = Image.open(row["mask_path"]).convert("L")
    _save(mask, p["mask"], N)
    return 1


if __name__ == "__main__":
    cfg = Config()
    manifest = MergedDatasetSplitter(cfg).run()     # scrive manifest_split.csv
    os.makedirs(CACHE, exist_ok=True)
    rows = manifest.to_dict("records")
    with Pool(os.cpu_count() // 2) as pool:
        done = list(tqdm(pool.imap_unordered(_process, rows, chunksize=32), total=len(rows)))
    print(f"Cache pronta: {sum(done)} nuove coppie, {len(rows) - sum(done)} già presenti.")