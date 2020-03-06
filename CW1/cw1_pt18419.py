import sys
import numpy as np
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from utilities import load_points_from_file


def setify20(xs, ys):
    xSets = []
    ySets = []
    for i in range(0, len(xs)//20):
        xSets.append(xs[i*20:i*20+20])
        ySets.append(ys[i * 20:i * 20 + 20])
    return np.array(xSets), np.array(ySets)


def least_squares_linear(xs, ys):
    ones = np.ones(len(xs))
    xtrans = np.array((ones, xs))
    x = xtrans.T
    xxt = np.matmul(xtrans, x)
    a = np.linalg.inv(xxt) @ xtrans @ ys.T
    return a


def least_squares_cubic(xs, ys):
    ones = np.ones(len(xs))
    x = np.column_stack((ones, xs, xs**2, xs**3))
    xtrans = x.T
    a = np.linalg.inv(xtrans @ x) @ xtrans @ ys.T
    return a


def least_squares_sin(xs, ys):
    ones = np.ones(len(xs))
    x = np.column_stack((ones, np.sin(xs)))
    xtrans = x.T
    a = np.linalg.inv(xtrans @ x) @ xtrans @ ys.T
    return a


def cal_linear_cross(xs, ys, xs_test, y_test):
    a = least_squares_linear(xs, ys)
    ds = a[0] + a[1] * xs_test
    cv = ((y_test - ds) ** 2).mean()
    return cv, a


def cal_cubic_cross(xs, ys, xs_test, y_test):
    a = least_squares_cubic(xs, ys)
    ds = a[0] + a[1] * xs_test + a[2] * xs_test**2 + a[3] * xs_test**3
    cv = ((y_test - ds) ** 2).mean()
    return cv, a


def cal_sin_cross(xs, ys, xs_test, y_test):
    a = least_squares_sin(xs, ys)
    ds = a[0] + a[1] * np.sin(xs_test)
    cv = ((y_test - ds) ** 2).mean()
    return cv, a


# def cal_linear(xs, ys):
#     a = least_squares_linear(xs, ys)
#     ds = a[0] + a[1] * xs
#     residual = 0
#     for j in range(0, len(xs)):
#         residual += ((ys)[j] - ds[j]) ** 2
#     return residual, ds
#
#
# def cal_cubic(xs, ys):
#     a = least_squares_cubic(xs, ys)
#     ds = a[0] + a[1] * xs + a[2] * xs**2 + a[3] * xs**3
#     residual = 0
#     for j in range(0, len(xs)):
#         residual += ((ys)[j] - ds[j]) ** 2
#     return residual, ds
#
#
# def cal_sin(xs, ys):
#     a = least_squares_sin(xs, ys)
#     ds = a[0] + a[1] * np.sin(xs)
#     residual = 0
#     for j in range(0, len(xs)):
#         residual += ((ys)[j] - ds[j]) ** 2
#     return residual, ds


####################################################################################################
####################################################################################################


# loading points from file
xs, ys = load_points_from_file(str(sys.argv[1]))
# grouping the sets into arrays of 20
xSets, ySets = setify20(xs, ys)

residual = 0

# iterating over the number of sets
for i in range(0, len(xSets)):

    plt.scatter(xSets[i], ySets[i])

    # trying test and split

    # X_train, X_test, y_train, y_test = train_test_split(xSets[i], ySets[i], test_size=0.2)
    # Yh_test = cal_linear_cross(X_train, y_train, X_test)
    # cross_validation_error = ((y_test - Yh_test) ** 2).mean()

    # trying K fold cross validation

    # minimum splits is 2

    number_of_splits = 20

    kf = KFold(n_splits=number_of_splits, shuffle=True)
    kf.get_n_splits(xSets[i])

    cross_validation_error_kfold_linear = 0
    average_linear = np.array([0.0, 0.0])
    for train_index, test_index in kf.split(xSets[i]):
        X_train, X_test = (xSets[i])[train_index], (xSets[i])[test_index]
        y_train, y_test = (ySets[i])[train_index], (ySets[i])[test_index]
        cv, a = cal_linear_cross(X_train, y_train, X_test, y_test)
        cross_validation_error_kfold_linear += cv
        average_linear += a

    average_linear = average_linear/number_of_splits
    cross_validation_error_kfold_linear = cross_validation_error_kfold_linear/number_of_splits

    cross_validation_error_kfold_cubic = 0
    average_cubic = np.array([0.0, 0.0, 0.0, 0.0])
    for train_index, test_index in kf.split(xSets[i]):
        X_train, X_test = (xSets[i])[train_index], (xSets[i])[test_index]
        y_train, y_test = (ySets[i])[train_index], (ySets[i])[test_index]
        cv, a = cal_cubic_cross(X_train, y_train, X_test, y_test)
        cross_validation_error_kfold_cubic += cv
        average_cubic += a

    average_cubic = average_cubic / number_of_splits
    cross_validation_error_kfold_cubic = cross_validation_error_kfold_cubic / number_of_splits

    cross_validation_error_kfold_sin = 0
    average_sin = np.array([0.0, 0.0])
    for train_index, test_index in kf.split(xSets[i]):
        X_train, X_test = (xSets[i])[train_index], (xSets[i])[test_index]
        y_train, y_test = (ySets[i])[train_index], (ySets[i])[test_index]
        cv, a = cal_sin_cross(X_train, y_train, X_test, y_test)
        cross_validation_error_kfold_sin += cv
        average_sin += a

    average_sin = average_sin / number_of_splits
    cross_validation_error_kfold_sin = cross_validation_error_kfold_sin / number_of_splits

    cross_validation_error_kfold = min(cross_validation_error_kfold_linear,
                                       cross_validation_error_kfold_cubic,
                                       cross_validation_error_kfold_sin)

    if cross_validation_error_kfold == cross_validation_error_kfold_linear:
        ds = average_linear[0] + average_linear[1] * xSets[i]
        for j in range(0, len(xSets[i])):
            residual += ((ySets[i])[j] - ds[j]) ** 2
        plt.plot(xSets[i], ds, c='blue')
    elif cross_validation_error_kfold == cross_validation_error_kfold_cubic:
        ds = average_cubic[0] + average_cubic[1] * xSets[i] +\
             average_cubic[2] * xSets[i]**2 + average_cubic[3] * xSets[i]**3
        for j in range(0, len(xSets[i])):
            residual += ((ySets[i])[j] - ds[j]) ** 2
        plt.plot(xSets[i], ds, c='green')
    elif cross_validation_error_kfold == cross_validation_error_kfold_sin:
        ds = average_sin[0] + average_sin[1] * np.sin(xSets[i])
        for j in range(0, len(xSets[i])):
            residual += ((ySets[i])[j] - ds[j]) ** 2
        plt.plot(xSets[i], ds, c='red')

    # trying Leave One Out Cross Validation

    # loo = LeaveOneOut()
    # loo.get_n_splits(xSets[i])
    # cross_validation_error_loocv = 0
    # for train_index, test_index in loo.split(xSets[i]):
    #     X_train, X_test = (xSets[i])[train_index], (xSets[i])[test_index]
    #     y_train, y_test = (ySets[i])[train_index], (ySets[i])[test_index]
    #     Yh_test = cal_linear_cross(X_train, y_train, X_test, y_test)
    #     cross_validation_error_loocv += ((y_test - Yh_test) ** 2).mean()
    #
    # cross_validation_error_loocv = cross_validation_error_loocv / number_of_splits

    # using only residuals

    # residual_lin, ds_lin = cal_linear(xSets[i], ySets[i])
    # residual_qub, ds_cub = cal_cubic(xSets[i], ySets[i])
    # residual_sin, ds_sin = cal_sin(xSets[i], ySets[i])
    # residual = min(residual_lin, residual_qub, residual_sin)
    #
    # if residual == residual_lin:
    #     ds = ds_lin
    #     plt.plot(xSets[i], ds, c='blue')
    # elif residual == residual_cub:
    #     ds = ds_qub
    #     plt.plot(xSets[i], ds, c='green')
    # elif residual == residual_sin:
    #     ds = ds_sin
    #     plt.plot(xSets[i], ds, c='red')

print(residual)

# adding legend to the plot

plt.plot(xSets[0][0], ySets[0][0], c='blue', label='linear')
plt.plot(xSets[0][0], ySets[0][0], c='green', label='polynomial')
plt.plot(xSets[0][0], ySets[0][0], c='red', label='unknown')
plt.legend()
if len(sys.argv) == 3 and str(sys.argv[2]) == "--plot":
    plt.show()
