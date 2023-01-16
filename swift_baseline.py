# loading lib
### Libraries for Data Handling

from pathlib import Path
import numpy as np
import pandas as pd
import timeit

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

# dataset dir
SWIFT_DATA_DIR = "/datasets/PET/dev_swift_transaction_train_dataset.csv"
SWIFT_DATA_DIR_TEST = "/datasets/PET/dev_swift_transaction_test_dataset.csv"
BANK_DATA_DIR = "/datasets/PET/dev_bank_dataset.csv"
swift_train = pd.read_csv(SWIFT_DATA_DIR, index_col="MessageId")
swift_train["Timestamp"] = swift_train["Timestamp"].astype("datetime64[ns]")

# loading dataset into panda frame
swift_test = pd.read_csv(SWIFT_DATA_DIR_TEST, index_col="MessageId")
swift_test["Timestamp"] = swift_test["Timestamp"].astype("datetime64[ns]")


# starting the pre-processing, onlyh keeping the useful columns
# preprocessing number 2
# Sender-Currency Frequency and Average Amount per Sender-Currency
starttime = timeit.default_timer()

swift_train["sender_currency"] = swift_train["Sender"] + swift_train["InstructedCurrency"]
swift_test["sender_currency"] = swift_test["Sender"] + swift_test["InstructedCurrency"]

sender_currency_freq = {}
sender_currency_avg = {}

for sc in set(
    list(swift_train["sender_currency"].unique()) + list(swift_test["sender_currency"].unique())
):
    sender_currency_freq[sc] = len(swift_train[swift_train["sender_currency"] == sc])
    sender_currency_avg[sc] = swift_train[swift_train["sender_currency"] == sc][
        "InstructedAmount"
    ].mean()

swift_train["sender_currency_freq"] = swift_train["sender_currency"].map(sender_currency_freq)
swift_test["sender_currency_freq"] = swift_test["sender_currency"].map(sender_currency_freq)

swift_train["sender_currency_amount_average"] = swift_train["sender_currency"].map(
    sender_currency_avg
)
swift_test["sender_currency_amount_average"] = swift_test["sender_currency"].map(sender_currency_avg)

print("The time difference is :", timeit.default_timer() - starttime)