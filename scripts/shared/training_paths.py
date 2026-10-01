from pathlib import Path
import os
from shared import train_cst
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import InstructTextDatasetType
from core.models.utility_models import ImageModelType

def get_checkpoints_output_path(task_id: str, repo_name: str) -> str:
    return str(Path(train_cst.OUTPUT_CHECKPOINTS_PATH) / task_id / repo_name)

def get_image_training_zip_save_path(task_id: str) -> str:
    return str(Path(train_cst.CACHE_DATASETS_DIR) / f'{task_id}.zip')

def get_text_dataset_path(task_id: str) -> str:
    return str(Path(train_cst.CACHE_DATASETS_DIR) / f'{task_id}_train_data.json')

def get_text_base_model_path(model_id: str) -> str:
    model_folder = model_id.replace('/', '--')
    return str(Path(train_cst.CACHE_MODELS_DIR) / model_folder)
