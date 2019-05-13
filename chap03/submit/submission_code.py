
x_train, y_train, x_test = load_fashionmnist()

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def np_log(x):
    return np.log(np.clip(a=x, a_min=1e-10, a_max=x))

def softmax(x):
    x -= x.max(axis=1, keepdims=True) # expのunderflow & overflowを防ぐ
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp, axis=1, keepdims=True)

# weights
W_fmnist = np.random.uniform(low=-0.08, high=0.08, size=(784, 10)).astype('float32')
b_fmnist = np.zeros(shape=(10,)).astype('float32')

# 学習データと検証データに分割
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1)

def train(x, t, eps=0.8):
    
    global W_fmnist, b_fmnist
    
    batch_size = x.shape[0]
    
    # 順伝播
    y = softmax(np.matmul(x, W_fmnist) + b_fmnist) # shape: (batch_size, 出力の次元数)
    
    # 逆伝播
    cost = (- t * np_log(y)).sum(axis=1).mean()
    delta = y - t # shape: (batch_size, 出力の次元数)
    
    # パラメータの更新
    dW = np.matmul(x.T, delta) / batch_size # shape: (入力の次元数, 出力の次元数)
    db = np.matmul(np.ones(shape=(batch_size,)), delta) / batch_size # shape: (出力の次元数,)
    W_fmnist -= eps * dW
    b_fmnist -= eps * db

    return cost

def valid(x):
    y = softmax(np.matmul(x, W_fmnist) + b_fmnist)
    return y

for epoch in range(1):
    for x, t in zip(x_train, y_train):
        cost = train(x[None, :], t[None, :])
    
y_pred = valid(x_valid).argmax(axis=1)

print('Valid Accuracy: {:.3f}'.format(accuracy_score(y_valid.argmax(axis=1), y_pred)))

submission = pd.Series(valid(x_test).argmax(axis=1), name='label')
submission.to_csv('/root/userspace/submission_pred.csv', header=True, index_label='id')
