from math import exp

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from prettytable import PrettyTable


if __name__ == "__main__":
    """Задание 1. Поиск шага интегрирования h (НАДО НАЙТИ ВО ВРЕМЯ ЗАНЯТИЯ!!!!!!!)"""
    print("\n\tЗадание 1. Найти шаг интегрирования h\n")

    eps = 0.0001
    h = np.sqrt(np.sqrt(eps))
    print("Шаг интегрирования", h)

    a = 1; b = 2; y_0 = 1
    X = np.arange(a, b + h, h)
    func = lambda x, y: (y**3 * exp(-x) - y) / x

    #-----------------------------------------------------------------------------------------

    """Задание 2. Поиск решения задачи Коши на отрезке [a,b] методом Рунге-Кутта"""

    Y_Cauchy = [y_0]
    for i,x in enumerate(X):
        F1 = func(x, Y_Cauchy[i])
        F2 = func(x + h/2, Y_Cauchy[i] + h*F1/2)
        F3 = func(x + h/2, Y_Cauchy[i] + h*F2/2)
        F4 = func(x + h, Y_Cauchy[i] + h*F3)
        Y_Cauchy.append(Y_Cauchy[i] + h*(F1 + 2*F2 + 2*F3 + F4)/6)
    Y_Cauchy = Y_Cauchy[:-1]

    #-----------------------------------------------------------------------------------------

    """Задание 3. Поиск решения задачи Коши на отрезке [a,b] методом Эйлера"""

    Y_Euler = [y_0]
    for i,x in enumerate(X):
        Y_Euler.append(Y_Euler[i] + h*func(x, Y_Euler[i]))
    Y_Euler = Y_Euler[:-1]
    
    #-----------------------------------------------------------------------------------------

    """Задание 4. Поиск решения задачи Коши с помощью функций Python"""

    Y_Python = odeint(func, y_0, X)
    Y_Python = Y_Python.reshape((-1))

    #-----------------------------------------------------------------------------------------

    """Задание 5. Поиск отклонений в узловых точках"""
    print("\n\tЗадание 5. Поиск отклонений в узловых точках\n")

    table = PrettyTable()
    table.field_names = [
        "x",
        "Euler Method",
        "Runge-Kutta Method",
        "Pyhton Solution",
        "Euler Abs Error",
        "Runge-Kutta Abs Error"
    ]
    for i in range(len(X)):
        table.add_row(
            [X[i],
            Y_Euler[i],
            Y_Cauchy[i],
            Y_Python[i],
            np.abs(Y_Python[i] - Y_Cauchy[i]),
            np.abs(Y_Python[i] - Y_Euler[i])]
        )
    print(table)

    plt.plot(X, Y_Euler, label="Euler Method")
    plt.plot(X, Y_Cauchy, label="Runge-Kutta Method")
    plt.plot(X, Y_Python, label="Pyhton Solution")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
