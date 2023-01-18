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

    return swift_test 

def combine_swift_and_bank(swift_data, bank_data, if_test):
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
    #of_onehot = pd.get_dummies(combine['OrderingFlags'], prefix='OF')
    #combine = pd.merge(combine, of_onehot, left_index=True, right_index=True)

    #bf_onehot = pd.get_dummies(combine['BeneficiaryFlags'], prefix='BF')
    #combine = pd.merge(combine, bf_onehot, left_index=True, right_index=True)

    # not all flags exsits, so need to fill in that doesnt exist
    #for i in range(13):
    #    name_of  = 'OF_' + str(int(i)) + '.0'
    #    name_bf = 'BF_' + str(int(i)) + '.0'
    #    if name_of not in combine.columns:
    #        combine[name_of] = 0
    #    if name_bf not in combine.columns:
    #        combine[name_bf] = 0
    
    
    # rearrange
    #if if_test:
    #    cols = ['SettlementAmount', 'InstructedAmount', 'hour',
    #            'sender_hour_freq', 'currency_freq','currency_amount_average', 
    #            'receiver_currency_freq','receiver_currency_amount_average','sender_receiver_freq',
    #            'OF_0.0', 'OF_1.0', 'OF_2.0', 'OF_3.0', 'OF_4.0','OF_5.0', 'OF_6.0', 'OF_7.0', 'OF_8.0', 'OF_9.0', 'OF_10.0', 'OF_11.0', 'OF_12.0','BF_0.0','BF_1.0', 'BF_2.0', 'BF_3.0', 'BF_4.0', 'BF_5.0', 'BF_6.0', 'BF_7.0', 'BF_8.0',
    #            'BF_9.0', 'BF_10.0', 'BF_11.0','BF_12.0']
    #else:
    #    cols = ['SettlementAmount', 'InstructedAmount', 'Label', 'hour',
    #            'sender_hour_freq', 'currency_freq','currency_amount_average', 
    #            'receiver_currency_freq','receiver_currency_amount_average','sender_receiver_freq',
    #            'OF_0.0', 'OF_1.0', 'OF_2.0', 'OF_3.0', 'OF_4.0','OF_5.0', 'OF_6.0', 'OF_7.0', 'OF_8.0', 'OF_9.0', 'OF_10.0', 'OF_11.0', 'OF_12.0','BF_0.0','BF_1.0', 'BF_2.0', 'BF_3.0', 'BF_4.0', 'BF_5.0', 'BF_6.0', 'BF_7.0', 'BF_8.0',
    #            'BF_9.0', 'BF_10.0', 'BF_11.0','BF_12.0']
    if if_test:   
        cols = ['SettlementAmount', 'InstructedAmount','hour',
                    'sender_hour_freq', 'currency_freq','currency_amount_average', 
                    'receiver_currency_freq','receiver_currency_amount_average','sender_receiver_freq',
                    'OrderingFlags','BeneficiaryFlags']
    else:
        cols = ['SettlementAmount', 'InstructedAmount','Label', 'hour',
                    'sender_hour_freq', 'currency_freq','currency_amount_average', 
                    'receiver_currency_freq','receiver_currency_amount_average','sender_receiver_freq',
                    'OrderingFlags','BeneficiaryFlags']
    
    combine = combine[cols]


    return combine
def combine_swift_and_bank_new_test(swift_test, bank_train, need_label):
    # combine the table and add flag features columns
    combine_test = (
        swift_test.reset_index().rename(columns={'index': 'MessageId'})
        .merge(
            right=bank_train[["Bank", "Account", "Flags"]].rename(
                columns={"Flags": "OrderingFlags"}
            ),
            how="left",
            left_on=["OrderingAccount"],
            right_on=["Account"],
        )
        .set_index("MessageId")
    )

    combine_test = (
        combine_test.reset_index().rename(columns={'index': 'MessageId'})
        .merge(
            right=bank_train[["Bank", "Account", "Flags"]].rename(
                columns={"Flags": "BeneficiaryFlags"}
            ),
            how="left",
            left_on=["BeneficiaryAccount"],
            right_on=["Account"],
        )
        .set_index("MessageId")
    )

    # fill the nan in the flags to value 12 ??????
    #if not if_test:
    #combine_train['OrderingFlags'] = combine_train['OrderingFlags'].fillna(12)
    #combine_train['BeneficiaryFlags'] = combine_train['BeneficiaryFlags'].fillna(12)

    # one hot for the flags

    of_onehot = pd.get_dummies(combine_test['OrderingFlags'], prefix='OF')
    combine_test_after = pd.merge(combine_test, of_onehot, left_index=True, right_index=True)

    bf_onehot = pd.get_dummies(combine_test['BeneficiaryFlags'], prefix='BF')
    combine_test_after = pd.merge(combine_test_after, bf_onehot, left_index=True, right_index=True)

    # not all flags exsits, so need to fill in that doesnt exist

    for i in range(13):
        name_of  = 'OF_' + str(int(i)) + '.0'
        name_bf = 'BF_' + str(int(i)) + '.0'
        if name_of not in combine_test_after.columns:
            combine_test_after[name_of] = 0
        if name_bf not in combine_test_after.columns:
            combine_test_after[name_bf] = 0
    
    # drop 
    # drop the columns thats not useful in training XGBoost
    columns_to_drop = [
        "OrderingFlags",
        "BeneficiaryFlags"
    ]

    combine_test_after = combine_test_after.drop(columns_to_drop, axis=1)
    if need_label:
        cols = ['SettlementAmount', 'InstructedAmount', 'Label', 'hour',
        'sender_hour_freq', 'currency_freq','currency_amount_average', 'sender_receiver_freq', 'receiver_currency_amount_average','sender_receiver_freq','OF_0.0', 'OF_1.0', 'OF_2.0', 'OF_3.0', 'OF_4.0',
        'OF_5.0', 'OF_6.0', 'OF_7.0', 'OF_8.0', 'OF_9.0', 'OF_10.0', 'OF_11.0', 'OF_12.0','BF_0.0',
        'BF_1.0', 'BF_2.0', 'BF_3.0', 'BF_4.0', 'BF_5.0', 'BF_6.0', 'BF_7.0', 'BF_8.0',
        'BF_9.0', 'BF_10.0', 'BF_11.0','BF_12.0']
    else:
        cols = ['SettlementAmount', 'InstructedAmount', 'hour',
        'sender_hour_freq', 'currency_freq','currency_amount_average', 'sender_receiver_freq', 'receiver_currency_amount_average','sender_receiver_freq','OF_0.0', 'OF_1.0', 'OF_2.0', 'OF_3.0', 'OF_4.0',
        'OF_5.0', 'OF_6.0', 'OF_7.0', 'OF_8.0', 'OF_9.0', 'OF_10.0', 'OF_11.0', 'OF_12.0','BF_0.0',
        'BF_1.0', 'BF_2.0', 'BF_3.0', 'BF_4.0', 'BF_5.0', 'BF_6.0', 'BF_7.0', 'BF_8.0',
        'BF_9.0', 'BF_10.0', 'BF_11.0','BF_12.0']
    combine_test_after = combine_test_after[cols]

    return combine_test_after

def combine_swift_and_bank_new(swift_train, bank_train):
    # combine the table and add flag features columns
    combine_train = (
        swift_train.reset_index().rename(columns={'index': 'MessageId'})
        .merge(
            right=bank_train[["Bank", "Account", "Flags"]].rename(
                columns={"Flags": "OrderingFlags"}
            ),
            how="left",
            left_on=["OrderingAccount"],
            right_on=["Account"],
        )
        .set_index("MessageId")
    )


    combine_train = (
        combine_train.reset_index().rename(columns={'index': 'MessageId'})
        .merge(
            right=bank_train[["Bank", "Account", "Flags"]].rename(
                columns={"Flags": "BeneficiaryFlags"}
            ),
            how="left",
            left_on=["BeneficiaryAccount"],
            right_on=["Account"],
        )
        .set_index("MessageId")
    )


    # fill the nan in the flags to value 12 ??????
    #if not if_test:
    combine_train['OrderingFlags'] = combine_train['OrderingFlags'].fillna(12)
    combine_train['BeneficiaryFlags'] = combine_train['BeneficiaryFlags'].fillna(12)

    # one hot for the flags
    of_onehot = pd.get_dummies(combine_train['OrderingFlags'], prefix='OF')
    combine_after = pd.merge(combine_train, of_onehot, left_index=True, right_index=True)

    bf_onehot = pd.get_dummies(combine_train['BeneficiaryFlags'], prefix='BF')
    combine_after = pd.merge(combine_after, bf_onehot, left_index=True, right_index=True)


    # not all flags exsits, so need to fill in that doesnt exist
    for i in range(13):
        name_of  = 'OF_' + str(int(i)) + '.0'
        name_bf = 'BF_' + str(int(i)) + '.0'
        if name_of not in combine_after.columns:
            combine_after[name_of] = 0
        if name_bf not in combine_after.columns:
            combine_after[name_bf] = 0


    
    # drop 
    # drop the columns thats not useful in training XGBoost
    columns_to_drop = [
        "OrderingFlags",
        "BeneficiaryFlags"
    ]

    combine_after = combine_after.drop(columns_to_drop, axis=1)

    cols = ['SettlementAmount', 'InstructedAmount', 'Label', 'hour',
       'sender_hour_freq', 'currency_freq','currency_amount_average', 'sender_receiver_freq', 'receiver_currency_amount_average','sender_receiver_freq','OF_0.0', 'OF_1.0', 'OF_2.0', 'OF_3.0', 'OF_4.0',
       'OF_5.0', 'OF_6.0', 'OF_7.0', 'OF_8.0', 'OF_9.0', 'OF_10.0', 'OF_11.0', 'OF_12.0','BF_0.0',
       'BF_1.0', 'BF_2.0', 'BF_3.0', 'BF_4.0', 'BF_5.0', 'BF_6.0', 'BF_7.0', 'BF_8.0',
       'BF_9.0', 'BF_10.0', 'BF_11.0','BF_12.0']
    combine_after = combine_after[cols]


    return combine_after

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

def get_X_swift( X):
    X_swift = []
    for i in range(len(X)):
        X_swift.append(X[i][:-26])
    X_swift = np.asarray(X_swift)
    return X_swift

def get_X_logistic_regression(X, pred_proba_xgb):
    X_lg = []
    for idx in range(len(X)):
        temp = X[idx][-26:]
        temp = np.append(temp,pred_proba_xgb[idx])
        X_lg.append(temp)
    X_lg = np.asarray(X_lg)

    return X_lg

def transform_and_normalized_Y(combine):
    Y = combine["Label"].values
    return  Y

def get_trainloader_for_NN(X_lg, Y):
    set = TrainData(torch.FloatTensor(X_lg), torch.FloatTensor(Y))
    dataloader = DataLoader(set, batch_size=32)
    return dataloader

def get_testloader_for_NN(X_lg):
    set = TestData(torch.FloatTensor(X_lg))
    dataloader = DataLoader(set, batch_size=32, shuffle=False)
    return dataloader
    


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


def train_NN( model, train_loader, device):
        model.to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        avg_acc = 0
        avg_loss = 0
        total = 0

        model.train()
        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            target = target.unsqueeze(1)

            optimizer.zero_grad()
            output = model(data)

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
    
def test_NN(model, test_loader, device):
    model.to(device)
    y_proba_list = []
    model.eval()
    with torch.no_grad():
        for _, (data) in enumerate(test_loader):
            data = data.to(device)
            output = model(data)
            y_proba_list.extend(output.detach().cpu().numpy())
    return y_proba_list

