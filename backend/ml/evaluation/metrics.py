import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve


def summarize(labels, predictions, probabilities, timings) -> dict:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    both = len(set(labels)) == 2
    points = []
    if both:
        fpr, tpr, thresholds = roc_curve(labels, probabilities)
        for i in np.unique(np.linspace(0, len(fpr) - 1, min(100, len(fpr))).astype(int)):
            points.append(dict(fpr=float(fpr[i]), tpr=float(tpr[i])))
    return dict(metrics=dict(accuracy=float(accuracy_score(labels, predictions)),
        precision=float(precision_score(labels, predictions, zero_division=0)),
        recall=float(recall_score(labels, predictions, zero_division=0)),
        f1=float(f1_score(labels, predictions, zero_division=0)),
        roc_auc=float(roc_auc_score(labels, probabilities)) if both else None,
        fpr=float(fp/(fp+tn)) if fp+tn else None, fnr=float(fn/(fn+tp)) if fn+tp else None),
        confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]], roc_points=points,
        average_inference_time=float(np.mean(timings)), p95_inference_time=float(np.percentile(timings, 95)))
