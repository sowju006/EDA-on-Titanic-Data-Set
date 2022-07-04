from data import download
from features import extract_features

data = download.data_Download()
features, labels = extract_features.features_extract(data)


from sklearn.model_selection import train_test_split


def split(features, labels):
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels)
    return X_train, X_test, Y_train, Y_test

def model():
    pass
