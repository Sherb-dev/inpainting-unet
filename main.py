# Base
import os
import torch
import random
import numpy as np
import pandas as pd
 
 # Torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import torch.nn as nn
 
# Custom Imports
from src.loss import *
from src.config import Config
from src.trainer import Trainer
from src.unet import AttentionUNet, PlainUNet
from src.evaluator import Evaluator
from src.preprocessing import JointTransform, InpaintingDataset
from src.dataset_builder import Dataset
from build_cache import build_cache

 
def set_seed(seed: int, dett_mode = False):
    # Fissa tutti i semi random per rendere gli esperimenti riproducibili
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # per una riproducibilità elevata
    if dett_mode:
        # Massima riproducibilità (più lento):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Mantiene la GPU al massimo della velocità:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    # Inizializza il generatore casuale di ciascun sottoprocesso worker
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Con model intendiamo l'architettura (UNet classica o con attenzione)
def main(model_name: str, cfg: Config, model: nn.Module, disable_compression_test=False):

    # ---------- IMPOSTAZIONE DEVICE E MULTI-GPU ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print(f"Utilizzo di {torch.cuda.device_count()} GPU abilitato.")
        model = nn.DataParallel(model)

    model = model.to(device)

    # Generatore PyTorch dedicato per il DataLoader (controlla lo shuffling del Sampler)
    g = torch.Generator()
    g.manual_seed(cfg.SEED)

    manifest = pd.read_csv(cfg.MANIFEST_OUT) # il manifest viene generato da build_cache.py

    # ---------- ADDESTRAMENTO ----------
    # trasformiamo le liste in oggetti dataset, ridimensioniamo immagini e maschere e nel caso di train=True facciamo data augmentation
    train_ds = InpaintingDataset(manifest, "train", JointTransform(cfg.IMAGE_SIZE, train=True, disable_compression_test=True), cache_dir=cfg.CACHE_DIR)
    val_ds = InpaintingDataset(manifest, "val", JointTransform(cfg.IMAGE_SIZE, train=False), cache_dir=cfg.CACHE_DIR)
    test_ds = InpaintingDataset(manifest, "test", JointTransform(cfg.IMAGE_SIZE, train=False), cache_dir=cfg.CACHE_DIR)


    # suddivisione in batch con shuffle solo nel training set
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=cfg.NUM_WORKERS > 0,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=cfg.NUM_WORKERS > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=cfg.NUM_WORKERS > 0,
    )

    # training (stessa procedura per entrambe le reti)
    Trainer(model=model, cfg=cfg, model_name=model_name).fit(train_loader, val_loader)


if __name__ == "__main__":

    BASE_DIR = Config.BASE_DIR
    IMAGE_SIZE = Config.IMAGE_SIZE

    cfg_bagel = Config()
    cfg_bagel.CACHE_DIR = f"{BASE_DIR}/dataset/cache_bagel_ex1_{IMAGE_SIZE}"
    cfg_bagel.MANIFEST_OUT = f"{BASE_DIR}/dataset/manifest_split_bagel_ex1.csv"
    
    """
    cfg_flux = Config()
    cfg_flux.CACHE_DIR = f"{BASE_DIR}/dataset/cache_flux_{IMAGE_SIZE}"
    cfg_flux.MANIFEST_OUT = f"{BASE_DIR}/dataset/manifest_split_flux.csv"

    cfg_sensenova = Config()
    cfg_sensenova.CACHE_DIR = f"{BASE_DIR}/dataset/cache_sensenova_{IMAGE_SIZE}"
    cfg_sensenova.MANIFEST_OUT = f"{BASE_DIR}/dataset/manifest_split_sensenova.csv"
    """
    
    bagel = Dataset(
                    original_images_path="/workspace/Test/BAGEL_Simple",
                    manip_images_path="/workspace/Datasets/BAGEL_hdpainter",
                    mask_path_list="/workspace/Masks/BAGEL"
    )
    build_cache(cfg_bagel, [bagel])
    
    main(
        model_name="Unet_Att_bagel_exp1",
        cfg=cfg_bagel,
        model=AttentionUNet(),
        disable_compression_test=True,
    )

    """
    main(
        model_name="Unet_Att_flux",
        cfg=cfg_flux,
        model=AttentionUNet(),
        disable_compression_test=True,
    )

    main(
        model_name="Unet_Att_sensenova",
        cfg=cfg_sensenova,
        model=AttentionUNet(),
        disable_compression_test=True,
    )
    """
