import numpy as np


def fit_cycle(model, X_train, X_test, Y_train, Y_test,
              epochs=100,
              callbacks=(),
              metrics=(),
              verbose=False):
    assert isinstance(metrics, tuple)

    progress_data = np.full((epochs + 1, len(metrics)), np.nan)
    Y_pred = np.full_like(Y_test, np.nan)

    for callback in callbacks:
        callback(model, X_test, Y_test, 0)

    if verbose:
        print(f'Before training')
    for j, (x, y) in enumerate(zip(X_test, Y_test)):
        Y_pred[j] = model.predict(x)
    for k, (metric_name, metric) in enumerate(metrics):
        progress_data[0, k] = metric(Y_pred, Y_test)
        if verbose:
            print(f'metric {metric_name} = {progress_data[0, k]}')
    if verbose:
        print(f'-----\n')

    for i in range(epochs):
        # train cycle
        for x, y in zip(X_train, Y_train):
            model.backward(x, y)

        # test_cycle
        for j, (x, y) in enumerate(zip(X_test, Y_test)):
            Y_pred[j] = model.predict(x)
        if verbose:
            print(f'Epoch {i + 1}')
        for k, (metric_name, metric) in enumerate(metrics):
            progress_data[i+1, k] = metric(Y_pred, Y_test)
            if verbose:
                print(f'metric {metric_name} = {progress_data[i+1, k]}')
        if verbose:
            print(f'-----\n')

        for callback in callbacks:
            callback(model, X_test, Y_test, i + 1)

    return progress_data

