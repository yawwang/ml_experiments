
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import metrics
import seaborn as sns

def evaluate_perf(predicted, gt, model_name):
    acc = metrics.accuracy_score(gt,predicted)
    bal_acc = metrics.balanced_accuracy_score(gt,predicted)
    precision_score = metrics.precision_score(gt,predicted)
    recall_score = metrics.recall_score(gt,predicted)
    print('Accuracy: {}\nBalanced Accuracy: {}\nPrecision: {}\nRecall: {}'.format(acc,bal_acc,precision_score,recall_score))
    print('Confusion Matrix:\n{}'.format(metrics.confusion_matrix(gt,predicted)))
    plot_roc(gt,predicted,model_name)
    
def plot_loss_history(history, start=None, end=None):
    if start == None: start=1
    if end == None: end=len(history.history['loss'])
    assert start<end
    assert end<=len(history.history['loss'])
    # Get training and test loss histories
    training_loss = history.history['loss'][start:end]
    test_loss = history.history['val_loss'][start:end]

    # Create count of the number of epochs
    epoch_count = range(start, end)

    # Visualize loss history
    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, test_loss, 'b-')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show();


def plot_roc(y,y_pred,name):
    fpr, tpr, threshold = metrics.roc_curve(y, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    fig, ax = plt.subplots(1, figsize=(12, 6))
    plt.plot(fpr, tpr, color='darkorange', label = 'AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', label='Random Performace')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('New York ROC Curve {}'.format(name))
    plt.legend(loc="lower right")


def tsplot(y, title, lags=None, figsize=(10, 6)):
    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0))
    hist_ax = plt.subplot2grid(layout, (0, 1))
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))

    y.plot(ax=ts_ax)
    ts_ax.set_title(title, fontsize=12, fontweight='bold')
    y.plot(ax=hist_ax, kind='hist', bins=25)
    hist_ax.set_title('Histogram')
    sm.graphics.tsa.plot_acf(y, lags=lags, ax=acf_ax)
    sm.graphics.tsa.plot_pacf(y, lags=lags, ax=pacf_ax)
    sns.despine()
    plt.tight_layout()
    plt.show()
    return ts_ax, acf_ax, pacf_ax