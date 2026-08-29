from imblearn.over_sampling import SMOTE
import pandas as pd


def get_balance_dataset(data, column, random, columns_list):
    """
    Applies SMOTE oversampling to balance dataset classes.
    """
    y_train = data[column]
    x_train = data.drop(column, axis=1)
    smote = SMOTE(random_state=random, k_neighbors=3)
    try:
        x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
    except Exception as e:
        print(f"Oversampling fallback: {e}")
        x_train_resampled, y_train_resampled = x_train, y_train

    x_train_resampled = pd.DataFrame(x_train_resampled, columns=x_train.columns)
    y_train_resampled = pd.DataFrame(y_train_resampled, columns=y_train.to_frame().columns)
    data_sampled = pd.concat([x_train_resampled, y_train_resampled], axis=1)
    data_sampled = data_sampled[columns_list]
    return data_sampled
