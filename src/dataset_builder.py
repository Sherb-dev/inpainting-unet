import glob
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision.transforms.functional as TF

from src.config import Config

class Dataset:
    def __init__(self, original_images_path:str, manip_images_path:str, mask_path_list:str):
        self.original_images_path = original_images_path
        self.manip_images_path = manip_images_path
        self.mask_path_list = mask_path_list

    def __iter__(self):
        yield self.original_images_path
        yield self.manip_images_path
        yield self.mask_path_list


class DatasetBuilder:
    """
    Crea un dataset unico dati n dataset in input.
    Richiede un file dataset.csv nella cartella manip con header:
    ["first_level_file", "original_class", "second_level_file", "prompt", "resolution", "dilatation_factor"]
    """
    def __init__(
        self,
        datasets: list[Dataset],
        cfg: Config,
    ):
        self._dataset = datasets
        self._cfg = cfg

    def _get_mask_info(self, file_name: str, masks_path: str, mask_extension: str):
        base_name = os.path.splitext(file_name)[0]
        pattern_ricerca = os.path.join(masks_path, f"{base_name}_*.{mask_extension}")
        matches = glob.glob(pattern_ricerca)

        if not matches:
            return "", "", ""

        mask_path = matches[0]
        mask_filename = os.path.basename(mask_path)
        nome_no_ext = os.path.splitext(mask_filename)[0]
        mask_class = nome_no_ext.split("_")[-1]

        return mask_filename, mask_class, mask_path

    def _get_image_id(self, file_name: str) -> str:
        """Estrae l'id: 0000_bagel.png -> 0000"""
        return os.path.splitext(file_name)[0].split("_")[0]

    def _verify_no_leakage(self, manifest: pd.DataFrame):
        splits_per_id = manifest.groupby("image_id")["split"].nunique()
        leaking = splits_per_id[splits_per_id > 1]
        if len(leaking) > 0:
            raise ValueError(f"Leakage rilevato su {len(leaking)} image_id.")
        print("Verifica leakage: OK.")

    @staticmethod
    def _report_missing(missing: list, name: str):
        if missing:
            print(f"[{name}] ATTENZIONE: {len(missing)} file mancanti:")
            for m in missing[:5]:
                print(f"  - {m}")
            raise FileNotFoundError(f"[{name}] {len(missing)} file mancanti.")

    def _init_manifest(self) -> pd.DataFrame:
        header = [
            "image_id", "orig_path", "manip_path", "mask_path",
            "category_name", "target_prompt", "source", "mask_type"
        ]
        return pd.DataFrame(columns=header)

    def _pre_build_dataset(
        self,
        dataset_original_path: str,
        dataset_manip_path: str,
        dataset_masks_path: str,
        mask_extension: str = "npy",
    ) -> pd.DataFrame:
        dataset_path = os.path.join(dataset_manip_path, "dataset.csv")
        assert os.path.exists(dataset_path), f"dataset.csv non trovato in: {dataset_manip_path}"

        df = pd.read_csv(dataset_path)
        source_name = Path(dataset_original_path).name

        rows, missing = [], []

        for _, r in df.iterrows():
            orig_filename = str(r["first_level_file"])
            manip_filename = str(r["second_level_file"])

            original_path = os.path.join(dataset_original_path, orig_filename)
            manip_path = os.path.join(dataset_manip_path, manip_filename)
            _, _, mask_path = self._get_mask_info(
                file_name=orig_filename,
                masks_path=dataset_masks_path,
                mask_extension=mask_extension,
            )
            image_id = self._get_image_id(manip_filename)

            # Verifica esistenza fisica
            for p in (original_path, manip_path, mask_path):
                if not p or not os.path.exists(p):
                    missing.append(p if p else f"Mask for {orig_filename}")

            rows.append({
                "image_id": f"{source_name}_{image_id}",
                "orig_path": original_path,
                "manip_path": manip_path,
                "mask_path": mask_path,
                "category_name": r.get("original_class", ""),
                "target_prompt": r.get("prompt", ""),
                "source": source_name,
                "mask_type": mask_extension,
                "dilatation_factor": r["dilatation_factor"] # edited
            })

        self._report_missing(missing, source_name)
        return pd.DataFrame(rows)

    def _build_dataset(self) -> pd.DataFrame:
        manifests_list = []
        # Corretto l'accesso alla variabile: si usa la lunghezza della lista dei Dataset passata nell'init
        totale = len(self._dataset)

        # Corretto l'iteratore: iteriamo direttamente su self._dataset 
        # sfruttando la funzione generatrice __iter__ definita nella classe Dataset
        for i, (orig, manip, mask) in enumerate(self._dataset, start=1):
            print(f"+ [{i}/{totale}] Processo {Path(orig).name}")
            dataset = self._pre_build_dataset(
                dataset_original_path=orig,
                dataset_manip_path=manip,
                dataset_masks_path=mask,
            )
            manifests_list.append(dataset)

        manifest = self._init_manifest()
        if manifests_list:
            manifest = pd.concat([manifest] + manifests_list, ignore_index=True)

        return manifest

    def _split_by_image_id(self, manifest: pd.DataFrame) -> pd.DataFrame:
        id_source = manifest.drop_duplicates("image_id")[["image_id", "source"]].set_index("image_id")["source"]
        unique_ids = id_source.index.to_numpy()
        strata = id_source.to_numpy()

        train_ids, rest_ids, _, rest_strata = train_test_split(
            unique_ids,
            strata,
            train_size=self._cfg.TRAIN_FRAC,
            random_state=self._cfg.SEED,
            stratify=strata,
        )

        relative_val = self._cfg.VAL_FRAC / (self._cfg.VAL_FRAC + self._cfg.TEST_FRAC)
        val_ids, test_ids = train_test_split(
            rest_ids,
            train_size=relative_val,
            random_state=self._cfg.SEED,
            stratify=rest_strata,
        )

        id_to_split = (
            {i: "train" for i in train_ids}
            | {i: "val" for i in val_ids}
            | {i: "test" for i in test_ids}
        )

        manifest = manifest.copy()
        manifest["split"] = manifest["image_id"].map(id_to_split)
        return manifest

    def run(self) -> pd.DataFrame:
        print("=== MERGE E SPLIT DEL DATASET ===")

        print("\n[1/3] Costruzione manifest...")
        manifest = self._build_dataset()

        # Undersampling bilanciato opzionale
        if getattr(self._cfg, "DATASET_SIZE", None) is not None:
            print(f"\n+ [UNDERSAMPLING] Riduzione a {self._cfg.DATASET_SIZE} campioni per source...")
            parts = []
            for _, group in manifest.groupby("source"):
                n = min(self._cfg.DATASET_SIZE, len(group))
                parts.append(group.sample(n=n, random_state=self._cfg.SEED))
            manifest = pd.concat(parts, ignore_index=True)
            print(f" - {len(manifest)} coppie totali dopo undersampling")

        print("\n+ Controllo duplicati...")
        dupes = manifest["image_id"].duplicated()
        if dupes.any():
            print(f"  ATTENZIONE: {dupes.sum()} image_id duplicati:")
            print(manifest[dupes][["image_id", "source"]].to_string())

        print("\n[2/3] Split dataset...")
        print(f"+ Split ratio: Train={self._cfg.TRAIN_FRAC} | Val={self._cfg.VAL_FRAC} | Test={self._cfg.TEST_FRAC}")
        manifest = self._split_by_image_id(manifest)
        self._verify_no_leakage(manifest)

        print("\n[3/3] Distribuzione per split:")
        dist = manifest.groupby(["split", "source"]).size().unstack(fill_value=0)
        print(dist)

        print("\nProporzioni relative per split:")
        print(dist.div(dist.sum(axis=1), axis=0).round(3))

        os.makedirs(os.path.dirname(self._cfg.MANIFEST_OUT), exist_ok=True)
        manifest.to_csv(self._cfg.MANIFEST_OUT, index=False)
        print(f"\n[COMPLETATO] Manifest salvato in: {self._cfg.MANIFEST_OUT}")

        return manifest