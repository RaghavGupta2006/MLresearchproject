# from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTE
import pandas as pd


def get_balance_dataset(data, column, random, columns_list):
    # print("过采样---------y", len(data[column]))
    # print("过采样---------x", len(data.drop(column, axis=1)))
    y_train = data[column]
    x_train = data.drop(column, axis=1)
    smote = SMOTE(random_state=random, k_neighbors=3)
    try:
        x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
    except:
        print("except sample")
        x_train_resampled, y_train_resampled = x_train, y_train

    x_train_resampled = pd.DataFrame(x_train_resampled, columns=x_train.columns)
    y_train_resampled = pd.DataFrame(y_train_resampled, columns=y_train.to_frame().columns)
    data_sampled = pd.concat([x_train_resampled, y_train_resampled], axis=1)
    data_sampled = data_sampled[columns_list]
    # print("过采样---------data_sampled", data_sampled.info())
    # print("过采样---------data_sampled", data_sampled)
    # print(data_sampled.isnull().sum())
    return data_sampled
