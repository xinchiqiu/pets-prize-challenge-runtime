from pathlib import Path
import joblib
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
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

class centralized_model:
    def __init__(self):
        self.xgb = XGBClassifier(n_estimators=100, max_depth = 7, base_score=0.01)
        self.lg =  LogisticRegression(random_state = 0)

    def pre_process_swift(self, swift_data):
        # Hour
        swift_data["hour"] = swift_data["Timestamp"].dt.hour

        # Hour frequency for each sender
        senders = swift_data["Sender"].unique()
        self.swift_data["sender_hour"] = swift_data["Sender"] + swift_data["hour"].astype(str)
        sender_hour_frequency = {}
        for s in senders:
            sender_rows = swift_data[swift_data["Sender"] == s]
            for h in range(24):
                sender_hour_frequency[s + str(h)] = len(sender_rows[sender_rows["hour"] == h])

        swift_data["sender_hour_freq"] = swift_data["sender_hour"].map(sender_hour_frequency)

        # Sender-Currency Frequency and Average Amount per Sender-Currency
        swift_data["sender_currency"] = swift_data["Sender"] + swift_data["InstructedCurrency"]

        sender_currency_freq = {}
        sender_currency_avg = {}

        for sc in set(
            list(swift_data["sender_currency"].unique()) #+ list(swift_test["sender_currency"].unique())
        ):
            sender_currency_freq[sc] = len(swift_data[swift_data["sender_currency"] == sc])
            sender_currency_avg[sc] = swift_data[swift_data["sender_currency"] == sc][
                "InstructedAmount"
            ].mean()

        swift_data["sender_currency_freq"] = swift_data["sender_currency"].map(sender_currency_freq)

        swift_data["sender_currency_amount_average"] = swift_data["sender_currency"].map(
            sender_currency_avg
        )

        # Sender-Receiver Frequency
        swift_data["sender_receiver"] = swift_data["Sender"] + swift_data["Receiver"]

        sender_receiver_freq = {}

        for sr in set(
            list(swift_data["sender_receiver"].unique()) #+ list(swift_test["sender_receiver"].unique())
        ):
            sender_receiver_freq[sr] = len(swift_data[swift_data["sender_receiver"] == sr])

        swift_data["sender_receiver_freq"] = swift_data["sender_receiver"].map(sender_receiver_freq)

        return swift_data
    
    def combine_swift_and_bank(self, swift_data, bank_data):
        # combine the table and add flag features columns
        combine = (
        swift_data.reset_index().rename(columns={'index': 'MessageId'})
            .merge(
                right=bank_data[["Bank", "Account", "Flags"]].rename(
                    columns={"Flags": "OrderingFlags"}
                ),
                how="left",
                left_on=["OrderingAccount"],
                right_on=["Account"],
            )
            .set_index("MessageId")
        )
        combine = (
        combine.reset_index().rename(columns={'index': 'MessageId'})
            .merge(
                right=bank_data[["Bank", "Account", "Flags"]].rename(
                    columns={"Flags": "BeneficiaryFlags"}
                ),
                how="left",
                left_on=["BeneficiaryAccount"],
                right_on=["Account"],
            )
            .set_index("MessageId")
        )

        # drop the columns thats not useful in training XGBoost
        columns_to_drop = [
            "UETR",
            "Sender",
            "Receiver",
            "TransactionReference",
            "OrderingAccount",
            "OrderingName",
            "OrderingStreet",
            "OrderingCountryCityZip",
            "BeneficiaryAccount",
            "BeneficiaryName",
            "BeneficiaryStreet",
            "BeneficiaryCountryCityZip",
            "SettlementDate",
            "SettlementCurrency",
            "InstructedCurrency",
            "Timestamp",
            "sender_hour",
            "sender_currency",
            "sender_receiver",
            "Bank_x",
            "Bank_y",
            "Account_y",
            "Account_x",
        ]

        combine = combine.drop(columns_to_drop, axis=1)
        return combine

    def transform_and_normalized(self, combine):
        Y = combine["Label"].values
        X = combine.drop(["Label"], axis=1).values

        # Normalize
        scaler = StandardScaler()
        scaler.fit(X)

        X = scaler.transform(X)
        return X, Y
    
    def get_X_swift(self, X):
        X_swift = []
        for i in range(len(X)):
            X_swift.append(X[i][:-2])
        X_swift = np.asarray(X_swift)
        return X_swift
    
    def get_X_logistic_regression(self,X, pred_proba_xgb):
        X_lg = []
        for idx in range(len(X)):
            temp = X[idx][-2:]
            temp = np.append(temp,pred_proba_xgb[idx])
            X_lg.append(temp)
        X_lg = np.asarray(X_lg)
        return X_lg
    

    def save(self, path):
        xgb_path = path / "centralized_xgboost.joblib"
        lg_path = path / "centralized_lg.joblib"
        joblib.dump(self.xgb, xgb_path)
        joblib.dump(self.lg, lg_path)
    

    @classmethod
    def load(cls, path):
        inst = cls()
        #inst.pipeline = joblib.load(path)
        xgb_path = path / "centralized_xgboost.joblib"
        lg_path = path / "centralized_lg.joblib"
        inst.xgb = joblib.load(xgb_path)
        inst.lg = joblib.load(lg_path)
        return inst


    

    


        