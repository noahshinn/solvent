import sys
import matplotlib.pyplot as plt

assert len(sys.argv) >= 3
LOG_FILES= sys.argv[1:-1]
SAVE_FILE = sys.argv[-1]

TITLE = 'NAC 4-bin training'


def get_value(line: str) -> float:
    return float(line.split(': ')[-1].replace('\n', ''))

fig, axs = plt.subplots(len(LOG_FILES) * 2)
fig.suptitle(TITLE)
fig.tight_layout()
for i, log_file in enumerate(LOG_FILES):
    train_mae = []
    train_mse = []
    test_mae = []
    test_mse = []
    cur_best_mae = 0.0
    cur_best_mse = 0.0
    with open(log_file, 'r') as f:
        data = f.readlines()
        for j, line in enumerate(data):
            if 'Train MAE: ' in line:
                train_mae += [get_value(line)]
            elif 'Train MSE: ' in line:
                train_mse += [get_value(line)]
            elif 'Test MAE: ' in line:
                val = get_value(line)
                test_mae += [val]
                if val < cur_best_mae:
                    cur_best_mae = val
                    cur_best_mse = get_value(data[j + 2])
            elif 'Test MSE: ' in line:
                test_mse += [get_value(line)]

    assert len(train_mae) == len(train_mse) == len(test_mae) == len(test_mse)

    epochs = list(range(len(train_mae)))
    axs[i * 2].plot(epochs, train_mae, label='Train MAE', linestyle=':')
    axs[i * 2 + 1].plot(epochs, train_mse, label='Train MSE', linestyle=':')
    axs[i * 2].plot(epochs, test_mae, label='Test MAE', linestyle='-')
    axs[i * 2 + 1].plot(epochs, test_mse, label='Test MSE', linestyle='-')
    axs[i * 2].legend()
    axs[i * 2 + 1].legend()

    print(f'Best MAE: {cur_best_mae}')
    print(f'Best MSE: {cur_best_mse}')
plt.show()
fig.savefig(SAVE_FILE)
