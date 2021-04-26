from sklearn.preprocessing import KBinsDiscretizer

from main import Lorenz
import random
from sklearn.metrics import mean_squared_error as RMSE

from sklearn.svm import SVR
from dim_work_version import delay_vector
import numpy as np
from sklearn import preprocessing


def generate_Z(mu, z_0, N):
    Z = [0] * N
    Z[0] = z_0
    for i in range(1, N):
        Z[i] = mu * Z[i - 1] * (1 - Z[i - 1])
    return Z


def preprocess_intervals(intervals):
    for y in intervals:
        tmp1 = np.log(y[0])
        tmp2 = np.log(y[1])
        y[0] = tmp1
        y[1] = tmp2
    return intervals


def inverse_transform_x_final(x_final):
    for i in range(len(x_final)):
        x_final[i] = np.exp(x_final[i])
    return x_final


discretizer = KBinsDiscretizer(n_bins=1000, encode='ordinal', strategy='uniform')
normalize_coef = 0


def preprocess_data(data):
    data = data.reshape((len(data), 1))

    global discretizer, normalize_coef
    data_discr = discretizer.fit_transform(data)
    normalize_coef = max(data_discr)

    preprocessed_data = preprocessing.normalize(data_discr, axis=0, norm='max')
    preprocessed_data = preprocessed_data.reshape(len(preprocessed_data), )
    return preprocessed_data


def recovery(data):
    global discretizer, normalize_coef
    for x in data:
        x *= normalize_coef
    recovered_data = discretizer.inverse_transform(data)
    return recovered_data


tau, dim, N = 15, 3, 1500

arr, t = Lorenz.solve(2 ** 13)  # решаем систему ЛОренца
data = arr[0][5000:7500]  # берем первую координату x
data = preprocess_data(data)
data_1 = data[:1500]
data_2 = data[1500:2500]
X = delay_vector(data_1, tau, dim)
data_1 = data_1[:-(dim - 1) * tau]
X_2 = delay_vector(data_2, tau, dim)
data_2 = data_2[:-(dim - 1) * tau]

intervals = preprocess_intervals(
    [[2 ** (-5), 2 ** 15], [2 ** (-12), 1], [2 ** (-15), 2 ** 5]])  # нормализованные интервалы для искомых констант

arr_Z = generate_Z(3.6, 0.2, N)


def compute_f(x):
    gamma = np.exp(x[2])
    C = np.exp(x[0])
    epsilon = np.exp(x[1])
    regr = SVR(kernel='rbf', gamma=gamma, C=C, epsilon=epsilon)
    regr.fit(X, data_1)
    x_pred = regr.predict(X)
    x_true = data_1
    return RMSE(x_true, x_pred) ** (1 / 2)


def step1():
    f_final, x_final = 0, [0] * 3
    for i in range(len(x_final)):
        z_i = random.choice(arr_Z)
        a = intervals[i][0]
        b = intervals[i][1]
        x_final[i] = a + (b - a) * z_i
    f_final = compute_f(x_final)
    return x_final, f_final


def step_2(z_i, intervals):
    arr = []
    for i in range(len(intervals)):
        a = intervals[i][0]
        b = intervals[i][1]
        arr.append(a + (b - a) * z_i)
    return arr


def step_3(x, x_final, f_final, A, time):
    f_tmp = compute_f(x)
    if f_tmp <= f_final:
        f_final = f_tmp
        x_final = x
        time += 1
    if time > A:
        return [True, x_final, f_final, time]
    else:
        return [False, x_final, f_final, time]


def chaos(A=10, M=10 ** 4, time=0):
    x_final, f_final = step1()
    flag = False
    K = 0
    J_end = 10e-4

    while not flag:
        z_i = random.choice(arr_Z)
        x = step_2(z_i, intervals)
        arr = step_3(x, x_final, f_final, A, time)
        flag, x_final, f_final, time = arr[0], arr[1], arr[2], arr[3]

        if K > M or f_final < J_end:
            return inverse_transform_x_final(x_final), f_final
        else:
            K += 1
            for i in range(len(intervals)):
                a = intervals[i][0]
                b = intervals[i][1]
                intervals[i][0] = x_final[i] - (b - a) / (K + 1)
                intervals[i][1] = x_final[i] + (b - a) / (K + 1)
                if intervals[i][0] < a:
                    intervals[i][0] = a
                if intervals[i][1] > b:
                    intervals[i][1] = b

    return inverse_transform_x_final(x_final), f_final


constants, rmse = chaos()
print('получившиеся константы', constants)
print('RMSE получилось на тренировочных данных', rmse)
model = SVR(kernel='rbf', gamma=constants[2], C=constants[0], epsilon=constants[1])
model.fit(X, data_1)
x_pred = model.predict(X_2)
x_true = data_2

print("RMSE на тестовых данных", RMSE(x_true, x_pred) ** (1 / 2))
