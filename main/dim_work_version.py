import numpy as np
from sklearn.neighbors import KDTree
import Lorenz


def Euqclid_distance(dot_1, dot_2):
    dist = 0
    for i in range(len(dot_1)):
        tmp = (dot_1[i] - dot_2[i]) ** 2
        dist += tmp
    return np.sqrt(dist)


def calculate_R_A(vector):
    N = len(vector)
    R_A_val = 0
    for i in range(1, N):
        tmp_1 = (1 / N) * sum(vector[i - 1:])
        R_A_val += (vector[i] - tmp_1) ** 2
    return np.sqrt(R_A_val / N)


# вектор с задержкой tau  в пространстве размерности dim
def delay_vector(vector, tau, dim):
    delay_array = []
    if dim == 1:
        vector = vector.reshape(-1, 1)
        return vector
    else:
        for i in range(dim):
            arr = np.roll(vector, -i * tau)[:-(dim - 1) * tau]
            delay_array.append(arr)
        X = []
        for i in range(len(delay_array[0])):
            tmp = []
            for j in range(len(delay_array)):
                tmp.append(delay_array[j][i])
            X.append(tmp)
        return X


def false_neighbors(vector_1, vector_2, RA, ATOL=10, RTOL=2):
    n = len(vector_1)

    kdt_1 = KDTree(vector_1, metric='euclidean')
    dist_neighbs_1 = kdt_1.query(vector_1, k=2, return_distance=True)
    dist_1 = dist_neighbs_1[0]
    neighbs_1 = dist_neighbs_1[1]
    dist_2 = [0] * n  # расстояния между соседями в пространстве m + 1

    # вычисляем расстояние между точками в пространстве m + 1
    for i in range(n):
        index = neighbs_1[i][1]
        dist = Euqclid_distance(vector_2[i], vector_2[index])
        dist_2[i] = dist

    # проверяем выполнения соотношений Кеннела
    false_count = 0
    for i in range(n):
        tmp = (dist_2[i] ** 2 - dist_1[i][1] ** 2) / dist_1[i][1] ** 2
        val_1 = np.sqrt(tmp)
        val_2 = dist_2[i] / RA
        if val_1 > RTOL or val_2 > ATOL:
            false_count += 1

    return false_count


# метод вычисления размерности вложенного пространства для
# данного вектора значений системы, шага = tau размерности оригинальног опространства
def false_nearest_neighbors(vector, tau, space_dim):
    R_tol = 15
    A_tol = 2
    R_A = calculate_R_A(vector)
    max_dim = 2 * space_dim + 1

    count = 10
    m = 1

    while count > 1e-2 and m <= max_dim:
        x = delay_vector(vector, tau, m)
        y = delay_vector(vector, tau, m + 1)
        x = x[:len(y)]

        M = len(x)
        count = false_neighbors(x, y, R_A, ATOL=A_tol, RTOL=R_tol) / M

        m += 1
    m -= 1

    return m
