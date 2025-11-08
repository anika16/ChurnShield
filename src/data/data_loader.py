from pathlib import Path
import pandas as pd
from loguru import logger
from ..config import RAW_FILE

def load_telco_data(path: str = None) -> pd.DataFrame:
    path = path or RAW_FILE
    logger.info(f"Loading dataset from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded dataset with shape {df.shape}")
    return df
