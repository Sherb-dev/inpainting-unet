# ==============================================================================
# 0. CONFIGURAZIONE E RIPRODUCIBILITÀ
# ==============================================================================
import torch
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent # punta all directory base

    # -- TEST ---
    DATA_DIR = None

    # --- Percorsi output ---
    MANIFEST_OUT = f"/{BASE_DIR}/dataset/manifest_split.csv"
    CHECKPOINT_DIR = f"{BASE_DIR}/outputs/checkpoints"
    FIGURES_DIR = f"{BASE_DIR}/outputs/figures"

    # --- Undersampling ----
    # None per non eseguirlo, il seed di sampling utilizzato è su varie
    DATASET_SIZE = None


    # --- Split ---
    TRAIN_FRAC = 0.80
    VAL_FRAC = 0.10
    TEST_FRAC = 0.10


    # --- Preprocessing ---
    IMAGE_SIZE = 256


    # --- Training ---
    BATCH_SIZE = 16
    EPOCHS = 120
    LR = 1e-4 # LR iniziale
    BASE_CHANNELS = 64
    NUM_WORKERS = 0
    

    # --- Early Stopping ---
    ES_PATIENCE  = 14    # usiamo un numeri alti in quanto con il cosine il lr scende lentamente
    ES_MIN_DELTA = 1e-3 # miglioramento minimo considerato significativo


    # --- Post-processing ---
    # + THRESHOLD
    SEG_THRESHOLD = 0.15 # Threshold di segmentazione, usato per la IoU
    CLS_THRESHOLD = 0.5 # Threshold di classificazione binaria 0.5

    # + MORFOLOGICO
    MORPHOLOGICAL_POSTPROCESS = False
    CLOSE_SIZE = 7
    MIN_AREA = 200
    # # # # # # # # # # # # # # 


    # --- Varie ---
    SEED = 42 # Influenza tutto ciò che ha bisogno di un seed
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # # # # # # # # 

    # METODI INCAPSULATI
    def set_new_data(self, new_data_dir:str, csv_name:str):
        import os
        self.DATA_DIR = new_data_dir
        self.CSV_PATH = os.path.join(new_data_dir, csv_name)

    def debug_paths(self):
        print(self.DATA_DIR)
        print(self.CSV_PATH)