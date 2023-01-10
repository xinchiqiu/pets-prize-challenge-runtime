from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd
from .solution_centralized import fit, predict


SWIFT_DATA_DIR = "/datasets/PET/dev_swift_transaction_train_dataset.csv"
SWIFT_DATA_DIR_TEST = "/datasets/PET/dev_swift_transaction_test_dataset.csv"
BANK_DATA_DIR = "/datasets/PET/dev_bank_dataset.csv"

fit(swift_data_path=SWIFT_DATA_DIR, bank_data_path = BANK_DATA_DIR, model_dir = "/model/centralized")