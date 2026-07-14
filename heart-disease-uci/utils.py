import pickle

def load_preprocessed_ds():

    # Load preprocessed dataset
    with open("heart_disease_processed.pkl", "rb") as f:
        X_train_pd, X_test_pd, y_train_pd, y_test_pd, scaler = pickle.load(f)

    # Convert dataset series/dataframes to numpy arrays
    X_train = X_train_pd.to_numpy()
    y_train = y_train_pd.to_numpy()
    X_test = X_test_pd.to_numpy()
    y_test = y_test_pd.to_numpy()

    return X_train, y_train, X_test, y_test, X_train_pd