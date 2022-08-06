import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# 载入数据集
def load_iris():
    iris_frame = pd.read_csv("iris.csv")
    X = np.c_[iris_frame["petal length"],iris_frame["petal width"]]
    y = np.asarray(iris_frame["is iris virginica"])
    return X, y

class CustomLinearSVC():
    def __init__(self):
        self.w = None
        self.b = None

    def _clip(self, alpha, L, H):
        ''' 修剪alpha的值到L和H之间.
        '''
        if alpha < L:
            return L
        elif alpha > H:
            return H
        else:
            return alpha

    def _select_j(self, i, m):
        ''' 在m中随机选择除了i之外剩余的数
        '''
        import random
        l = list(range(m))
        seq = l[: i] + l[i+1:]
        return random.choice(seq)

    def _simple_smo(self, dataset, labels, C=1, max_iter=500):
        ''' 简化版SMO算法实现，未使用启发式方法对alpha对进行选择.
        :param dataset: 所有特征数据向量
        :param labels: 所有的数据标签
        :param C: 软间隔常数, 0 <= alpha_i <= C
        :param max_iter: 外层循环最大迭代次数
        '''
        dataset = np.array(dataset)
        m, n = dataset.shape
        labels = np.array(labels)
        # 初始化参数S
        alphas = np.zeros(m)
        b = 0
        it = 0

        def f(x):
            "SVM分类器函数 y = w^Tx + b"
            # Kernel function vector.
            x = np.matrix(x).T
            data = np.matrix(dataset)
            ks = data*x
            # Predictive value.
            wx = np.matrix(alphas*labels)*ks
            fx = wx + b
            return fx[0, 0]

        while it < max_iter:
            pair_changed = 0
            for i in range(m):
                a_i, x_i, y_i = alphas[i], dataset[i], labels[i]
                fx_i = f(x_i)
                E_i = fx_i - y_i
                j = self._select_j(i, m)
                a_j, x_j, y_j = alphas[j], dataset[j], labels[j]
                fx_j = f(x_j)
                E_j = fx_j - y_j
                K_ii, K_jj, K_ij = np.dot(x_i, x_i), np.dot(
                    x_j, x_j), np.dot(x_i, x_j)
                eta = K_ii + K_jj - 2*K_ij
                if eta <= 0:
                    # print('WARNING  eta <= 0')
                    continue
                # 获取更新的alpha对
                a_i_old, a_j_old = a_i, a_j
                a_j_new = a_j_old + y_j*(E_i - E_j)/eta
                # 对alpha进行修剪
                if y_i != y_j:
                    L = max(0, a_j_old - a_i_old)
                    H = min(C, C + a_j_old - a_i_old)
                else:
                    L = max(0, a_i_old + a_j_old - C)
                    H = min(C, a_j_old + a_i_old)
                a_j_new = self._clip(a_j_new, L, H)
                a_i_new = a_i_old + y_i*y_j*(a_j_old - a_j_new)
                if abs(a_j_new - a_j_old) < 0.00001:
                    #print('WARNING   alpha_j not moving enough')
                    continue
                alphas[i], alphas[j] = a_i_new, a_j_new
                # 更新阈值b
                b_i = -E_i - y_i*K_ii*(a_i_new - a_i_old) - \
                    y_j*K_ij*(a_j_new - a_j_old) + b
                b_j = -E_j - y_i*K_ij*(a_i_new - a_i_old) - \
                    y_j*K_jj*(a_j_new - a_j_old) + b
                if 0 < a_i_new < C:
                    b = b_i
                elif 0 < a_j_new < C:
                    b = b_j
                else:
                    b = (b_i + b_j)/2
                pair_changed += 1
                # print('INFO   iteration:{}  i:{}  pair_changed:{}'.format(it, i, pair_changed))
            if pair_changed == 0:
                it += 1
            else:
                it = 0
            # print('iteration number: {}'.format(it))
        return alphas, b

    def train(self, X, y):
        '''使用输入数据X和标签y生成w和b'''
        alpha, b = self._simple_smo(X, y)
        t = np.asarray(y, dtype='int32')
        t[t == 0] = -1
        yx = t.reshape(1, -1).T*np.array([1, 1])*X
        self.w = np.dot(yx.T, alpha)
        self.b = b

    def evaluate(self, X):
        '''使用模型w^TX+b=0进行估算'''
        res = np.dot(X,self.w)+self.b
        res[res>0]=1
        res[res<0]=0
        return res


# 鸢尾花数据集
# 使用特征: Row #2 (花瓣长度) & Row #3 (花瓣宽度)
def plot_scatter(X, y):
    y_class = (y == 0).astype(np.int32)
    plt.scatter(X[:, 0][y_class == 1], X[:, 1][y_class == 1],
                c="#7C7CFF", label="Iris virginica")
    plt.scatter(X[:, 0][y_class == 0], X[:, 1][y_class == 0],
                c="#FF7C7C", label="Not iris virginica")
    plt.xlabel("Petal length (cm)")
    plt.ylabel("Petal width (cm)")
    plt.legend()

# 辅助：绘制决策边界
def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int(
            (axis[1]-axis[0])*200)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int(
            (axis[3]-axis[2])*200)).reshape(-1, 1)
    )
    # c_[]将两个数组以列的形式拼接起来，形成矩阵。
    xx = np.c_[x0.ravel(), x1.ravel()]
    y_predict = model.evaluate(xx)

    zz = y_predict.reshape(x0.shape)  # 通过训练好的模型，预测平面上这些点的分类

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#EEEEEE', '#90CAF9'])
    plt.contourf(x0, x1, zz, linewidths=2, linestyles='dashed',
                 cmap=custom_cmap, alpha=0.5)
    plt.contour(x0, x1, zz, linewidths=2, linestyles='solid', colors="#000000")


if __name__ == "__main__":
    X, y = load_iris()
    linear_svc = CustomLinearSVC()
    linear_svc.train(X, y == 0)
    plot_decision_boundary(linear_svc, [-0.2, 7.2, -0.1, 2.7])
    plot_scatter(X, y)
    plt.show()
