from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sklearn.utils
from .model import CentralizedModel

def fit(
    swift_data_path: Path,
    bank_data_path: Path,
    model_dir: Path,
) -> None:
    """Function that fits your model on the provided training data and saves
    your model to disk in the provided directory.

    Args:
        swift_data_path (Path): Path to CSV data file for the SWIFT transaction
            dataset.
        bank_data_path (Path): Path to CSV data file for the bank account
            dataset.
        model_dir (Path): Path to a directory that is constant between the train
            and test stages. You must use this directory to save and reload
            your trained model between the stages.
        preds_format_path (Path): Path to CSV file matching the format you must
            write your predictions with, filled with dummy values.
        preds_dest_path (Path): Destination path that you must write your test
            predictions to as a CSV file.

    Returns: None
    """
    # read the table
    swift_train_ori = pd.read_csv(swift_data_path, index_col="MessageId")
    bank_train = pd.read_csv(bank_data_path)

    swift_train_ori.to_pickle("swift_train.pkl")
    swift_train = pd.read_pickle("swift_train.pkl")
    swift_train["Timestamp"] = swift_train["Timestamp"].astype("datetime64[ns]")

    # init centralized model
    centralized_model = CentralizedModel()

    # pre-processing swift
    swift_train = centralized_model.pre_process_swift(swift_train)

    # combine with bank dataset for centralized training
    combine_train = centralized_model.combine_swift_and_bank(swift_train, bank_train)

    # get X_train and Y_train
    X_train, Y_train = centralized_model.transform_and_normalized(combine_train)

    # get trainset for XGBoost
    X_train_swift = centralized_model.get_X_swift(X_train)

    # fit for XGBoost
    centralized_model.xgb.fit(X_train_swift,Y_train)

    # get XGBoost prediction probability
    pred_proba_xgb_train = centralized_model.xgb.predict_proba(X_train_swift)[:, 1]

    # get trainset for logistic regression
    X_train_lg = centralized_model.get_X_logistic_regression(X_train, pred_proba_xgb_train)
    centralized_model.lg.fit(X_train_lg,Y_train)

    logger.info("...done fitting")
    centralized_model.save(model_dir)


def predict(
    swift_data_path: Path,
    bank_data_path: Path,
    model_dir: Path,
    preds_format_path: Path,
    preds_dest_path: Path,
) -> None:
    """Function that loads your model from the provided directory and performs
    inference on the provided test data. Predictions should match the provided
    format and be written to the provided destination path.

    Args:
        swift_data_path (Path): Path to CSV data file for the SWIFT transaction
            dataset.
        bank_data_path (Path): Path to CSV data file for the bank account
            dataset.
        model_dir (Path): Path to a directory that is constant between the train
            and test stages. You must use this directory to save and reload
            your trained model between the stages.
        preds_format_path (Path): Path to CSV file matching the format you must
            write your predictions with, filled with dummy values.
        preds_dest_path (Path): Destination path that you must write your test
            predictions to as a CSV file.

    Returns: None

    """
    # read the table
    swift_test_ori = pd.read_csv(swift_data_path, index_col="MessageId")
    bank_test = pd.read_csv(bank_data_path)

    swift_test_ori.to_pickle("swift_train.pkl")
    swift_test = pd.read_pickle("swift_train.pkl")
    swift_test["Timestamp"] = swift_test["Timestamp"].astype("datetime64[ns]")

    logger.info("Preparing test data...")

    # init centralized model
    centralized_model = centralized_model(swift_test, bank_test)

    # pre-processing swift
    swift_test = centralized_model.pre_process_swift(swift_test)

    # combine with bank dataset for centralized training
    combine_test = centralized_model.combine_swift_and_bank(swift_test, bank_test)

    # get X_train and Y_train
    X_test, Y_test = centralized_model.transform_and_normalized(combine_test)
    X_test_swift = centralized_model.get_X_swift(X_test)

    # loading models
    logger.info("Loading models...")
    centralized_model = CentralizedModel.load(model_dir)

    # get XGBoost prediction
    pred_proba_xgb_test = centralized_model.xgb.predict_proba(X_test_swift)[:, 1]
    
    # get trainset for logistic regression
    X_test_lg = centralized_model.get_X_logistic_regression(X_test, pred_proba_xgb_test)
    
    # get lg prediction, which one to submit below?
    #final_preds = centralized_model.lg.predict(X_test_lg) # this is get 0/1 prediction
    final_preds = centralized_model.lg.predict_proba(X_test_lg)[:, 1] #this is get the probability prediction

    # this part is from the example, I'm assuming this is what we want to submit
    preds_format_df = pd.read_csv(preds_format_path, index_col="MessageId")
    preds_format_df["Score"] = preds_format_df.index.map(final_preds)

    logger.info("Writing out test predictions...")
    preds_format_df.to_csv(preds_dest_path)
    logger.info("Done.")
