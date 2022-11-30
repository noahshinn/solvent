import sys
import numpy as np
import matplotlib.pyplot as plt

assert len(sys.argv) == 3
LOG_FILE = sys.argv[1]
SAVE_FILE = sys.argv[2]


def get_value(line: str) -> float:
    return float(line.split(': ')[-1].replace('\n', ''))

def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

train_acc = []
train_loss = []
test_acc = []
test_loss = []
with open(LOG_FILE, 'r') as f:
    data = f.readlines()
    for line in data:
        if 'Train accuracy: ' in line:
            train_acc += [get_value(line)]
        elif 'Train loss: ' in line:
            train_loss += [get_value(line)]
        elif 'Test accuracy: ' in line:
            test_acc += [get_value(line)]
        elif 'Test loss: ' in line:
            test_loss += [get_value(line)]

assert len(train_acc) == len(train_loss) == len(test_acc) == len(test_loss)
train_acc = np.asarray(train_acc)
train_loss = np.asarray(train_loss)
test_acc = np.asarray(test_acc)
test_loss = np.asarray(test_loss)

train_acc_filtered = train_acc[~is_outlier(train_acc)]
train_loss_filtered = train_loss[~is_outlier(train_loss)]
test_acc_filtered = test_acc[~is_outlier(test_acc)]
test_loss_filtered = test_loss[~is_outlier(test_loss)]

# assert len(train_acc_filtered) == len(train_loss_filtered) == len(test_acc_filtered) == len(test_loss_filtered)

epochs = list(range(len(train_acc_filtered)))

fig, axs = plt.subplots(2)
fig.suptitle('NAC 3-bin classification training')
axs[0].plot(list(range(len(train_acc_filtered))), train_acc_filtered, label='Train acc', linestyle=':')
axs[1].plot(list(range(len(train_loss_filtered))), train_loss_filtered, label='Train loss', linestyle=':')
axs[0].plot(list(range(len(test_acc_filtered))), test_acc_filtered, label='Test acc', linestyle='-')
axs[1].plot(list(range(len(test_loss_filtered))), test_loss_filtered, label='Test loss', linestyle='-')
axs[0].legend()
axs[1].legend()
plt.show()
fig.savefig(SAVE_FILE)
