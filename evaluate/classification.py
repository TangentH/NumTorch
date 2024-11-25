import numpy as np

def evaluate_multi_class(model, test_ds, return_failure_case=False):
    y_true = test_ds.labels
    y_pred = model(test_ds.data)
    y_pred = y_pred.view(np.ndarray)
    y_true = y_true.view(np.ndarray)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)
    confusion_matrix = np.zeros((y_true.max()+1, y_true.max()+1))
    for i in range(len(y_true)):
        confusion_matrix[y_true[i], y_pred[i]] += 1
    acc = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    if return_failure_case:
        failure_case = []
        for i in range(len(y_true)):
            if y_true[i] != y_pred[i]:
                failure_case.append((i, y_true[i], y_pred[i]))
        return confusion_matrix, acc, failure_case
    return confusion_matrix, acc
    

def evaluate(model, test_ds):
    y_true = test_ds.labels
    y_pred = model(test_ds.data)
    y_pred = y_pred.view(np.ndarray).reshape(-1)
    y_true = y_true.view(np.ndarray).reshape(-1)
    y_pred[y_pred>=0.5] = 1
    y_pred[y_pred!=1] = 0
    epsilon = 1e-10
    confusion_matrix = np.zeros((2, 2))
    for i in range(len(y_true)):
        confusion_matrix[int(y_true[i]), int(y_pred[i])] += 1
    acc = np.trace(confusion_matrix) / (np.sum(confusion_matrix) + epsilon)
    recall = confusion_matrix[1,1]/(confusion_matrix[1,0]+confusion_matrix[1,1]+epsilon)
    precision = confusion_matrix[1,1]/(confusion_matrix[0,1]+confusion_matrix[1,1]+epsilon)
    f1 = 2*(precision*recall)/(precision+recall+epsilon)
    return confusion_matrix, acc, recall, precision, f1

def evaluate_perceptron(model, test_ds):
    y_true = test_ds.labels
    y_pred = model(test_ds.data)
    y_pred = y_pred.view(np.ndarray).reshape(-1)
    y_true = y_true.view(np.ndarray).reshape(-1)
    y_true[y_true==-1] = 0
    y_pred[y_pred>=0] = 1
    y_pred[y_pred<0] = 0
    epsilon = 1e-10
    confusion_matrix = np.zeros((2, 2))
    for i in range(len(y_true)):
        confusion_matrix[int(y_true[i]), int(y_pred[i])] += 1
    acc = np.trace(confusion_matrix) / (np.sum(confusion_matrix) + epsilon)
    recall = confusion_matrix[1,1]/(confusion_matrix[1,0]+confusion_matrix[1,1]+epsilon)
    precision = confusion_matrix[1,1]/(confusion_matrix[0,1]+confusion_matrix[1,1]+epsilon)
    f1 = 2*(precision*recall)/(precision+recall+epsilon)
    return confusion_matrix, acc, recall, precision, f1