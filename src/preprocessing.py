# pyright: reportAssignmentType=false

import os
import numpy as np
import random
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

class JointTransform:
    def __init__(self, size, train=True, disable_compression_test:bool = False):
        self.size = size
        self.train = train
        self.disable_compression_test = disable_compression_test

    def __call__(self, image: Image.Image, mask: Image.Image):
        # 1. Resize iniziale
        image = TF.resize(image, (self.size, self.size), interpolation=TF.InterpolationMode.BILINEAR)
        mask  = TF.resize(mask,  (self.size, self.size), interpolation=TF.InterpolationMode.NEAREST)

        if self.train:
            # Flips geometrici sicuri (non introducono bordi neri)
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask  = TF.vflip(mask)

            # Simulazione perdita di risoluzione / compressione (SOLO sull'immagine)
            if random.random() > 0.5 and self.disable_compression_test == False:
                small_size = self.size // 2
                image = TF.resize(image, (small_size, small_size), interpolation=TF.InterpolationMode.BILINEAR)
                image = TF.resize(image, (self.size, self.size), interpolation=TF.InterpolationMode.BILINEAR)

            # Variazioni fotometriche (non toccano la maschera)
            image = TF.adjust_brightness(image, random.uniform(0.85, 1.15))
            image = TF.adjust_contrast(image,   random.uniform(0.85, 1.15))
            image = TF.adjust_saturation(image, random.uniform(0.85, 1.15))
            image = TF.adjust_hue(image,        random.uniform(-0.05, 0.05))

            if random.random() > 0.85:
                image = TF.rgb_to_grayscale(image, num_output_channels=3)

        # 2. Conversione a tensore
        image = TF.to_tensor(image)
        # Normalizzazione standard
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()

        return image, mask


#usiamo questa classe per recuperare i file dal disco quando sono necessari, partendo dal manifest
class InpaintingDataset(Dataset):
    def __init__(self, manifest_df, split, transform: JointTransform, cache_dir=None, disable_compression_test:bool = False):
        
        self.rows = manifest_df[manifest_df["split"] == split].reset_index(drop=True)
        self.transform = transform
        self.cache_dir = cache_dir 
        self.disable_compression_test = disable_compression_test

    def __len__(self):
        return len(self.rows) * 2 #perchè per ogni immagine abbiamo sia la versione originale che quella manipolata

    def __getitem__(self, idx):
        row = self.rows.iloc[idx // 2] #recuperiamo l'indice
        is_manipulated = (idx % 2 == 1) #se è dispari corrisponde a un'immagine manipolata, se è pari a una originale

        if self.cache_dir:                                                               # NUOVO blocco
            iid = row["image_id"]
            if is_manipulated:
                image = Image.open(f"{self.cache_dir}/{iid}_manip.png").convert("RGB")
                mask = Image.open(f"{self.cache_dir}/{iid}_mask.png").convert("L")
                label = 1.0
            else:
                image = Image.open(f"{self.cache_dir}/{iid}_orig.png").convert("RGB")
                mask = Image.new("L", image.size, 0)
                label = 0.0
        elif is_manipulated:
            # se è manipolata apre direttamente il file
            image = Image.open(row["manip_path"]).convert("RGB")
            
            ext = os.path.splitext(row["mask_path"])[1]
            #controlla l'estensione della maschera (se è numpy la converte sennò la carica direttamente)
            if ext == ".npy":
                mask = Image.fromarray((np.load(row["mask_path"])*255).astype(np.uint8)).convert('L')
            else:
                mask = Image.open(row["mask_path"]).convert("L")
            
            label = 1.0 #assegna l'etichetta
        else:
            #se è originale apre l'immagine e crea una maschera completamente nera sul momento
            image = Image.open(row["orig_path"]).convert("RGB")
            mask = Image.new("L", image.size, 0)  # maschera nera generata al volo
            label = 0.0

        image, mask = self.transform(image, mask, self.disable_compression_test) #passa l'immagine per il preprocessing e eventuale data augmentation
        return image, mask, torch.tensor(label, dtype=torch.float32)
