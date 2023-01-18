from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd
import torch
from .model_centralized import *


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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info("pre-processing the swift dataset")
    swift_train = pre_process_swift(swift_train, model_dir)

    logger.info("Combine with bank dataset")
    combine_train = combine_swift_and_bank_new(swift_train, bank_train)

    logger.info("Get X_train and Y_train")
    X_train = transform_and_normalized_X_train(combine_train, model_dir)
    Y_train = transform_and_normalized_Y(combine_train)

    logger.info("Fit SWIFT XGBoost")

    X_train_swift = get_X_swift(X_train)
    xgb = XGBClassifier(n_estimators=100, max_depth = 7, base_score=0.01, learning_rate = 0.1)
    xgb.fit(X_train_swift,Y_train)

    logger.info("Save XGBoost")
    xgb.save_model(os.path.join(model_dir,"centralized_xgb.json"))

    logger.info("Get probability from XGBoost")
    pred_proba_xgb_train = xgb.predict_proba(X_train_swift)[:, 1]
    
    logger.info("get the trainset and dataloader for NN")
    X_train_lg = get_X_logistic_regression(X_train, pred_proba_xgb_train)
    train_dataloader = get_trainloader_for_NN(X_train_lg, Y_train)

    logger.info("fit for the 1 layer neural network")
    model = Net_lg()
    _, _ = train_NN(model = model, train_loader =train_dataloader, device = device)

    logger.info("Save NN")
    torch.save(model, os.path.join(model_dir, "centralized_nn.pt"))

    logger.info("...done fitting")


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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    logger.info("Loading models...")
    xgb = XGBClassifier()
    xgb.load_model(os.path.join(model_dir,'centralized_xgb.json'))
    model = torch.load(os.path.join(model_dir, "centralized_nn.pt"))

    logger.info("pre-processing swift test")
    swift_test = pre_process_swift_test(swift_test, model_dir)

    logger.info("Combine datasets")
    combine_test = combine_swift_and_bank_new_test(
        swift_test = swift_test,
        bank_train=bank_test,
        need_label=False
    ) 
    
    logger.info("transform and normalized the datasets")
    X_test = transform_and_normalized_X_test(
        combine = combine_test, 
        model_dir = model_dir, 
        if_exist_y=False
    )
    X_test_swift = get_X_swift(X_test)
    #Y_test = transform_and_normalized_Y(combine_test)

    logger.info("Predict with XGBoost on swift dataset") 
    pred_proba_xgb_test = xgb.predict_proba(X_test_swift)[:, 1]

    # get trainset for logistic regression(NN)
    X_test_lg = get_X_logistic_regression(X_test, pred_proba_xgb_test)
    test_dataloader = get_testloader_for_NN(X_test_lg)
    
    logger.info("Predict with logistic regression or NN")
    preds = test_NN(model,test_dataloader,device)

    # convert to pandas series
    final_preds = pd.Series(preds, index=combine_test.index)

    # this part is from the example: to write the score
    preds_format_df = pd.read_csv(preds_format_path, index_col="MessageId")
    preds_format_df["Score"] = preds_format_df.index.map(final_preds)
    preds_format_df["Score"] = preds_format_df["Score"].astype(np.float64)

    logger.info("Writing out test predictions...")
    preds_format_df.to_csv(preds_dest_path)
    logger.info("Done.")
