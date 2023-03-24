from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# define functions
def prep_data(data):

    data = data.copy()

    X = data.drop("host", axis=1)
    y = data.pop("host")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return [X_train, X_test, y_train, y_test]


def create_baseline_model(X, hidden_units, dropout):

    input_dim = X.shape[1]
    input = keras.Input(shape=(input_dim,))
    x = input

    for units in hidden_units:
        x = layers.Dense(units)(x)
        #x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(dropout)(x)

    output = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=input, outputs=output)
    return model

def run_experiment(data, hidden_units, dropout=0.2, lr=0.001):

    X_train, X_test, y_train, y_test = prep_data(data)

    model = create_baseline_model(X_train, hidden_units, dropout)
    model.compile(Adam(learning_rate=lr), loss="binary_crossentropy", metrics=["AUC"])
    model.summary()

    history = model.fit(X_train, y_train, epochs=500, validation_split=0.2,
                        callbacks=[
                            EarlyStopping(monitor='val_loss', patience=10, mode='max', restore_best_weights=True)],
                        verbose=2)

    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.legend(['train', 'validation'], loc='upper left')

    _, auc = model.evaluate(X_test, y_test, verbose=0)

    print(f"Test AUC: {round(auc, 3)}")


def run_experiment_loo(data, hidden, dropout=0.2, lr=0.001):

    data = data.copy()

    X = data.drop("host", axis=1)
    y = data.pop("host")

    y_pred = []

    for i in range(X.shape[0]):
        X_train = X.drop(i)
        X_test = np.array(X.iloc[i]).reshape(-1,1).T
        y_train = y.drop(i)

        model = create_baseline_model(X_train, hidden, dropout)
        model.compile(Adam(learning_rate=lr), loss="binary_crossentropy")

        model.fit(X_train, y_train, epochs=500, validation_split=0.2,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=10, mode='max', restore_best_weights=True)])

        y_pred.append(model.predict(X_test))

    return np.concatenate(y_pred, axis=0)

