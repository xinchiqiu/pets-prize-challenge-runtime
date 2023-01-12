from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd
import torch
from .model_centralized import CentralizedModel


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

    logger.info("read the table")
    swift_train = pd.read_csv(swift_data_path, index_col="MessageId")
    bank_train = pd.read_csv(bank_data_path)
    swift_train["Timestamp"] = swift_train["Timestamp"].astype("datetime64[ns]")

    # init centralized model
    logger.info("initialized the model")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    centralized_model = CentralizedModel()

    logger.info("pre-processing the swift dataset")
    swift_train = centralized_model.pre_process_swift(swift_train)

    logger.info("Combine with bank dataset")
    combine_train = centralized_model.combine_swift_and_bank(swift_train, bank_train, False)

    logger.info("Get X_train and Y_train")
    X_train = centralized_model.transform_and_normalized_X(combine_train, False)
    Y_train = centralized_model.transform_and_normalized_Y(combine_train)

    logger.info("Fit SWIFT XGBoost")
    X_train_swift = centralized_model.get_X_swift(X_train)

    centralized_model.xgb.fit(X_train_swift,Y_train)

    logger.info("Get probability from XGBoost")
    pred_proba_xgb_train = centralized_model.xgb.predict_proba(X_train_swift)[:, 1]
    
    logger.info("get the trainset and dataloader for NN")
    X_train_lg = centralized_model.get_X_logistic_regression(X_train, pred_proba_xgb_train)
    train_dataloader = centralized_model.get_trainloader_for_NN(X_train_lg, Y_train)
    
    # choose one, either fit the logistic regression, or the 1 layer neural network
    #logger.info("fit for logistic regression")
    #centralized_model.lg.fit(X_train_lg,Y_train)

    logger.info("fit for the 1 layer neural network")
    _, _ = centralized_model.train_NN(train_loader =train_dataloader, device = device)

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
    logger.info("Preparing test data...")
    swift_test = pd.read_csv(swift_data_path, index_col="MessageId")
    bank_test = pd.read_csv(bank_data_path)
    swift_test["Timestamp"] = swift_test["Timestamp"].astype("datetime64[ns]")

    logger.info("Initialized the model")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    centralized_model = CentralizedModel()

    logger.info("pre-processing swift test")
    swift_test = centralized_model.pre_process_swift(swift_test)

    logger.info("Combine datasets")
    combine_test = centralized_model.combine_swift_and_bank(swift_test, bank_test, True)

    logger.info("transform and normalized the datasets")
    X_test = centralized_model.transform_and_normalized_X(combine_test, True)
    X_test_swift = centralized_model.get_X_swift(X_test)

    logger.info("Loading models...")
    centralized_model = CentralizedModel.load(model_dir)

    logger.info("Predict with XGBoost on swift dataset")
    pred_proba_xgb_test = centralized_model.xgb.predict_proba(X_test_swift)[:, 1]
    
    # get trainset for logistic regression
    X_test_lg = centralized_model.get_X_logistic_regression(X_test, pred_proba_xgb_test)
    test_dataloader = centralized_model.get_testloader_for_NN(X_test_lg)
    
    logger.info("Predict with logistic regression or NN")
    preds = centralized_model.test_NN(test_dataloader,device)
    #preds = centralized_model.lg.predict_proba(X_test_lg)[:, 1] #this is get the probability prediction

    # convert to pandas series
    final_preds = pd.Series(preds, index=combine_test.index)

    # this part is from the example: to write the score
    preds_format_df = pd.read_csv(preds_format_path, index_col="MessageId")
    preds_format_df["Score"] = preds_format_df.index.map(final_preds)

    logger.info("Writing out test predictions...")
    preds_format_df.to_csv(preds_dest_path)
    logger.info("Done.")

