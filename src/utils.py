import numpy as np
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef

def evaluate(model, X_test, y_test, encoder, name):
    pred = model.predict(X_test)
    pred_cls = np.argmax(pred, axis=1)

    acc = accuracy_score(y_test, pred_cls)
    mcc = matthews_corrcoef(y_test, pred_cls)

    print(f"\n{name} Accuracy:", acc)
    print(f"{name} MCC:", mcc)

    print(classification_report(
        y_test,
        pred_cls,
        target_names=encoder.classes_,
        zero_division=0
    ))

    return pred, acc

def ensemble(preds, weights):
    final = np.zeros_like(preds[0])
    for p, w in zip(preds, weights):
        final += w * p
    return final