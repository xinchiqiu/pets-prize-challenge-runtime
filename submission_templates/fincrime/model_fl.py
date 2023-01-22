from pathlib import Path
#import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn.pipeline import Pipeline
import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from xgboost import XGBClassifier

'''
so for swift client,
Need to do:
swift_train = pd.read_csv(swift_data_path, index_col="MessageId")
swift_train = pre_process_swift(swift_train, model_dir)
X_train = transform_and_normalized_X_train(swift_train, model_dir)
Y_train = transform_and_normalized_Y(swift_train)
model = SwiftModel()
model.fit(X_train,Y_train)
proba = model.predict_proba(X_train)
'''

class SwiftModel:
    def __init__(self):
        self.xgb = XGBClassifier()

    def fit(self, X_train, Y_train):
        self.xgb = XGBClassifier(n_estimators=100, max_depth = 7, base_score=0.01, learning_rate = 0.1)
        self.xgb.fit(X_train,Y_train)
        #return self

    def predict_proba(self, X):
        pred_proba_xgb_train = self.xgb.predict_proba(X)[:, 1]
        return pred_proba_xgb_train

    def save(self, path):
        self.xgb.save_model(os.path.join(path,"centralized_xgb.json"))

    @classmethod
    def load(cls, path):
        inst = cls()
        inst.xgb.load_model(os.path.join(path,'centralized_xgb.json'))
        return inst


def pre_process_swift(swift_train, model_dir):
    #pre-processing number 1
    # Hour frequency for each sender
    swift_train["hour"] = swift_train["Timestamp"].dt.hour
    senders = swift_train["Sender"].unique()
    swift_train["sender_hour"] = swift_train["Sender"] + swift_train["hour"].astype(str)
    sender_hour_frequency = {}
    for s in senders:
        sender_rows = swift_train[swift_train["Sender"] == s]
        for h in range(24):
            sender_hour_frequency[s + str(h)] = len(sender_rows[sender_rows["hour"] == h])
    swift_train["sender_hour_freq"] = swift_train["sender_hour"].map(sender_hour_frequency)
    
    #pre-processing number 2
    currency_freq = {}
    currency_avg = {}
    for sc in set(
        list(swift_train["InstructedCurrency"].unique())
    ):
        currency_freq[sc] = len(swift_train[swift_train["InstructedCurrency"] == sc])
        currency_avg[sc] = swift_train[swift_train["InstructedCurrency"] == sc][
            "InstructedAmount"
        ].mean()

    swift_train["currency_freq"] = swift_train["InstructedCurrency"].map(currency_freq)
    swift_train["currency_amount_average"] = swift_train["InstructedCurrency"].map(currency_avg)

    #pre-processing number 3
    swift_train["receiver_currency"] = swift_train["Receiver"] + swift_train["InstructedCurrency"]
    receiver_currency_freq = {}
    receiver_currency_avg = {}

    for sc in set(
        list(swift_train["receiver_currency"].unique())
    ):
        receiver_currency_freq[sc] = len(swift_train[swift_train["receiver_currency"] == sc])
        receiver_currency_avg[sc] = swift_train[swift_train["receiver_currency"] == sc][
            "InstructedAmount"
        ].mean()

    swift_train["receiver_currency_freq"] = swift_train["receiver_currency"].map(receiver_currency_freq)
    swift_train["receiver_currency_amount_average"] = swift_train["receiver_currency"].map(receiver_currency_avg)

    #pre-processing number 4
    #Sender-Receiver Frequency
    swift_train["sender_receiver"] = swift_train["Sender"] + swift_train["Receiver"]
    sender_receiver_freq = {}

    for sr in set(
        list(swift_train["sender_receiver"].unique())
    ):
        sender_receiver_freq[sr] = len(swift_train[swift_train["sender_receiver"] == sr])

    swift_train["sender_receiver_freq"] = swift_train["sender_receiver"].map(sender_receiver_freq)        

    # save the dictionary for test dataset
    with open(os.path.join(model_dir, 'sender_hour_frequency.pkl'), 'wb') as f:
        pickle.dump(sender_hour_frequency,f)
    with open(os.path.join(model_dir, 'currency_freq.pkl'), 'wb') as f:
        pickle.dump(currency_freq,f)
    with open(os.path.join(model_dir, 'currency_avg.pkl'), 'wb') as f:
        pickle.dump(currency_avg,f)
    with open(os.path.join(model_dir, 'receiver_currency_freq.pkl'), 'wb') as f:
        pickle.dump(receiver_currency_freq,f)
    with open(os.path.join(model_dir, 'receiver_currency_avg.pkl'), 'wb') as f:
        pickle.dump(receiver_currency_avg,f)
    with open(os.path.join(model_dir, 'sender_receiver_freq.pkl'), 'wb') as f:
        pickle.dump(sender_receiver_freq,f)

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
        "receiver_currency",
        "sender_receiver",
    ]

    swift_train = swift_train.drop(columns_to_drop, axis=1)

    return swift_train

def pre_process_swift_test(
        swift_test, 
        model_dir
    ):
    # load the dictionary for test dataset
    with open(os.path.join(model_dir, 'sender_hour_frequency.pkl'), 'rb') as f:
        sender_hour_frequency = pickle.load(f)
    with open(os.path.join(model_dir, 'currency_freq.pkl'), 'rb') as f:
        currency_freq = pickle.load(f)
    with open(os.path.join(model_dir, 'currency_avg.pkl'), 'rb') as f:
        currency_avg = pickle.load(f)
    with open(os.path.join(model_dir, 'receiver_currency_freq.pkl'), 'rb') as f:
        receiver_currency_freq = pickle.load(f)
    with open(os.path.join(model_dir, 'receiver_currency_avg.pkl'), 'rb') as f:
        receiver_currency_avg = pickle.load(f)
    with open(os.path.join(model_dir, 'sender_receiver_freq.pkl'), 'rb') as f:
        sender_receiver_freq = pickle.load(f)

    # number 1
    swift_test["hour"] = swift_test["Timestamp"].dt.hour
    swift_test["sender_hour"] = swift_test["Sender"] + swift_test["hour"].astype(str)
    swift_test["sender_hour_freq"] = swift_test["sender_hour"].map(sender_hour_frequency)

    # number 2
    swift_test["currency_freq"] = swift_test["InstructedCurrency"].map(currency_freq)
    swift_test["currency_amount_average"] = swift_test["InstructedCurrency"].map(currency_avg)

    # number 3
    swift_test["receiver_currency"] = swift_test["Receiver"] + swift_test["InstructedCurrency"]
    swift_test["receiver_currency_freq"] = swift_test["receiver_currency"].map(receiver_currency_freq)
    swift_test["receiver_currency_amount_average"] = swift_test["receiver_currency"].map(receiver_currency_avg)
    
    # number 4
    swift_test["sender_receiver"] = swift_test["Sender"] + swift_test["Receiver"]
    swift_test["sender_receiver_freq"] = swift_test["sender_receiver"].map(sender_receiver_freq)

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
        "receiver_currency",
        "sender_receiver",
    ]

    swift_test = swift_test.drop(columns_to_drop, axis=1)
    return swift_test 


def transform_and_normalized_X_train(combine,model_dir):
    X = combine.drop(["Label"], axis=1).values

    # Normalize
    scaler = StandardScaler()
    scaler.fit(X)

    X = scaler.transform(X)
    pickle.dump(scaler, open(os.path.join(model_dir,'scaler.pkl'),'wb'))
    return X

def transform_and_normalized_X_test(combine,model_dir,if_exist_y):
    if if_exist_y:
        X = combine.drop(["Label"], axis=1).values
    else:
        X = combine.values
    # load scaler
    scaler =pickle.load(open(os.path.join(model_dir,'scaler.pkl'),'rb'))
    X = scaler.transform(X)

    return X

def transform_and_normalized_Y(combine):
    Y = combine["Label"].values
    return  Y

if __name__ == "__main__":
    swift_data_path = '/datasets/PET/new_swift_transaction_train_dataset_dev/dev_swift_transaction_train_dataset.csv'
    swift_train = pd.read_csv(swift_data_path, index_col="MessageId")
    swift_train = pre_process_swift(swift_train, '/nfs-share/xinchi/pets-prize-challenge-runtime/model/FL')
    X_train = transform_and_normalized_X_train(swift_train, '/nfs-share/xinchi/pets-prize-challenge-runtime/model/FL')
    Y_train = transform_and_normalized_Y(swift_train)
    model = SwiftModel()
    model.fit(X_train,Y_train)
    proba = model.predict_proba(X_train)