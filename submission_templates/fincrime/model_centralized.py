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
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CentralizedModel:
    def __init__(self):
        self.xgb = XGBClassifier(n_estimators=100, max_depth = 7, base_score=0.01)
        #self.lg =  LogisticRegression(random_state = 0)
        self.nn = Net_lg()
        
        # dict saved for pre-processing the test set
        #self.sender_hour_frequency = {}
        #self.currency_freq = {}
        #self.currency_avg = {}

    def pre_process_swift(self, swift_train):
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

        return swift_train
    
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

        # fill the nan in the flags to value 12
        combine['OrderingFlags'] = combine['OrderingFlags'].fillna(12)
        combine['BeneficiaryFlags'] = combine['BeneficiaryFlags'].fillna(12)

        # one hot for the flags
        of_onehot = pd.get_dummies(combine['OrderingFlags'], prefix='OF')
        combine = pd.merge(combine, of_onehot, left_index=True, right_index=True)

        bf_onehot = pd.get_dummies(combine['BeneficiaryFlags'], prefix='BF')
        combine = pd.merge(combine, bf_onehot, left_index=True, right_index=True)

        # not all flags exsits, so need to fill in that doesnt exist
        for i in range(13):
            name_of  = 'OF_' + str(int(i)) + '.0'
            name_bf = 'BF_' + str(int(i)) + '.0'
            if name_of not in combine.columns:
                combine[name_of] = 0
            if name_bf not in combine.columns:
                combine[name_bf] = 0
        
        
        # rearrange
        cols = ['SettlementAmount', 'InstructedAmount', 'Label', 'hour',
                'sender_hour_freq', 'currency_freq','currency_amount_average', 
                'receiver_currency_freq','receiver_currency_amount_average','sender_receiver_freq',
                'OF_0.0', 'OF_1.0', 'OF_2.0', 'OF_3.0', 'OF_4.0','OF_5.0', 'OF_6.0', 'OF_7.0', 'OF_8.0', 'OF_9.0', 'OF_10.0', 'OF_11.0', 'OF_12.0','BF_0.0','BF_1.0', 'BF_2.0', 'BF_3.0', 'BF_4.0', 'BF_5.0', 'BF_6.0', 'BF_7.0', 'BF_8.0',
                'BF_9.0', 'BF_10.0', 'BF_11.0','BF_12.0']
        combine = combine[cols]

        return combine

    def transform_and_normalized_X(self, combine):
        X = combine.drop(["Label"], axis=1).values

        # Normalize
        scaler = StandardScaler()
        scaler.fit(X)

        X = scaler.transform(X)
        return X
    
    def transform_and_normalized_Y(self, combine):
        Y = combine["Label"].values
        return  Y

    def get_X_swift(self, X):
        X_swift = []
        for i in range(len(X)):
            X_swift.append(X[i][:-26])
        X_swift = np.asarray(X_swift)
        return X_swift
    
    def get_X_logistic_regression(self,X, pred_proba_xgb):
        X_lg = []
        for idx in range(len(X)):
            temp = X[idx][-26:]
            temp = np.append(temp,pred_proba_xgb[idx])
            X_lg.append(temp)
        X_lg = np.asarray(X_lg)
        X_lg = np.nan_to_num(X_lg, nan=12)
        return X_lg
    
    def get_trainloader_for_NN(self,X_lg, Y):
        set = TrainData(torch.FloatTensor(X_lg), torch.FloatTensor(Y))
        dataloader = DataLoader(set, batch_size=32)
        return dataloader

    def get_testloader_for_NN(self,X_lg):
        set = TestData(torch.FloatTensor(X_lg))
        dataloader = DataLoader(set, batch_size=32)
        return dataloader
    
    def train_NN(self, train_loader, device):
        self.nn.to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(self.nn.parameters(), lr=0.1)
        avg_acc = 0
        avg_loss = 0
        total = 0

        self.nn.train()
        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            target = target.unsqueeze(1)

            optimizer.zero_grad()
            output = self.nn(data)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # get statistics
            predicted = torch.round(output)
            correct = (predicted == target).sum()
            avg_acc += correct.item()
            avg_loss += loss.item() * data.shape[0]
            total += data.shape[0]

        return avg_acc/total, avg_loss/total
    
    def test_NN(self, test_loader, device):
        self.nn.to(device)
        criterion = nn.BCELoss()
        y_proba_list = []
        self.nn.eval()
        with torch.no_grad():
            for _, (data) in enumerate(test_loader):
                data = data.to(device)
                #target = target.unsqueeze(1)
                output = self.nn(data)
                #loss = criterion(output, target)

                # get statistics
                y_proba_list.extend(output.detach().cpu().numpy())
        return y_proba_list

    def save(self, path):
        xgb_path = os.path.join(path, "centralized_xgboost.pkl")
        #lg_path = os.path.join(path,"centralized_lg.pkl")
        nn_path = os.path.join(path,"centralized_nn.pkl")
        #joblib.dump(self.xgb, xgb_path)
        #joblib.dump(self.lg, lg_path)
        with open(xgb_path, 'wb') as f:
            pickle.dump(self.xgb, f)
        #with open(lg_path, 'wb') as f:
        #    pickle.dump(self.lg, f)  
        with open(nn_path, 'wb') as f:
            pickle.dump(self.nn, f) 
    

    @classmethod
    def load(cls, path):
        inst = cls()
        #inst.pipeline = joblib.load(path)
        xgb_path = os.path.join(path,"centralized_xgboost.pkl")
        #lg_path = os.path.join(path,"centralized_lg.pkl")
        nn_path = os.path.join(path,"centralized_nn.pkl")

        with open(xgb_path, 'rb') as f:
            inst.xgb = pickle.load(f)
        #with open(lg_path, 'rb') as f:
        #    inst.lg = pickle.load(f)
        with open(nn_path, 'rb') as f:
            inst.lg = pickle.load(f)
        return inst


class Net_lg(nn.Module):
    def __init__(self):
        super(Net_lg, self).__init__()
        self.layer_1 = nn.Linear(27, 1) 
        self.sigmoid =  nn.Sigmoid()
        
        
    def forward(self, inputs):
        x = self.sigmoid(self.layer_1(inputs))
        return x
    

class TrainData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)  

class TestData(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)  
