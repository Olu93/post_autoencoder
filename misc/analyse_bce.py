from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


def generate_loss_function(penalty=.5):
    assert penalty >= 0 and penalty <= 1, "Choose value between 0 and 1"

    def loss_function(y_true, y_pred):
        penalty_ = penalty * 100
        clipped_y_pred = np.clip(y_pred, 1e-2, 1.0 - 1e-2)
        binary_cross_entropy = -(penalty_ * y_true * np.log(clipped_y_pred) + (100 - penalty_) * (1.0 - y_true) * np.log(1.0 - clipped_y_pred)) / 100.0
        loss = np.sum(binary_cross_entropy)
        return loss, binary_cross_entropy

    return loss_function

def _rebase(x1, min_val, max_val):
    x1 = (x1 * (max_val - min_val)) + min_val
    return x1

func = generate_loss_function(.97)
data = np.load("data/dataset_voxels.npz")["data"]
X = data[0]
X_pred = np.ones_like(X)

X_l = X.flatten().tolist()
cnt = Counter(X_l)
plt.hist(X_l)
plt.title(str(cnt))
plt.savefig(f"misc/X_hist.png")
# plt.show()
print(cnt)

print("=============")
print("Only 1")
loss, bce = func(X, X_pred)
bce_l = bce.flatten().tolist()
cnt = Counter(bce_l)
plt.hist(bce_l)
plt.title(str(cnt))
plt.savefig(f"misc/bce_hist.png")
# plt.show()
print(cnt)
print(loss)

print("=============")
print("Only 0")
loss, bce = func(X, np.zeros_like(X_pred))
bce_l = bce.flatten().tolist()
cnt = Counter(bce_l)
plt.hist(bce_l)
plt.title(str(cnt))
plt.savefig(f"misc/bce_hist_zeros.png")
# plt.show()
print(cnt)
print(loss)

print("=============")
print("Only -1")
loss, bce = func(_rebase(X, -1, 2), np.ones_like(X_pred))
bce_l = bce.flatten().tolist()
cnt = Counter(bce_l)
plt.hist(bce_l)
plt.title(str(cnt))
plt.savefig(f"misc/bce_hist_neg.png")
# plt.show()
print(cnt)
print(loss)

print("=============")
print("Only same")
loss, bce = func(X, X)
bce_l = bce.flatten().tolist()
cnt = Counter(bce_l)
plt.hist(bce_l)
plt.title(str(cnt))
plt.savefig(f"misc/bce_hist_same.png")
# plt.show()
print(cnt)
print(loss)
