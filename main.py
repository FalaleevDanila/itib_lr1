from numpy import tanh
import matplotlib.pyplot as plt
import pylab
import itertools


def f1_(x):
    return 1


def f1(net):
    return 1 if net >= 0 else 0


def f4(net):
    return 0.5 * (tanh(net) + 1)


def f4_(x):
    return 0.5 - tanh(x)**2/2


def f_y(net):
    return 1 if net >= 0.5 else 0


def fun(x1, x2, x3, x4):
    return int(not ((x1 or x2) * x3 * x4))
    # return int(not(x1 and x2) and x3 and x4)


xs = [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [1, 1, 1, 1]
    ]


def print_data():
    true_fun = []
    for i in xs:
        x1, x2, x3, x4 = i
        true_fun.append(fun(x1, x2, x3, x4))
    print('<'*35)
    print('|| x1  | x2  | x3  | x4  |#|  F  ||')
    print('<'*35)
    for (i, j) in zip(xs, true_fun):
        print('|| ', i[0], ' | ', i[1], ' | ', i[2],
              ' | ', i[3], ' |#| ', j, ' ||')
        print('>'*35)


def print_diagram(x, y):
    pylab.xlabel('k')
    pylab.ylabel('E')
    plt.plot(x, y, 'ro-')
    plt.show()


class NeuralS:

    def f_net(self, x, w, w0):
        net = w0
        for (i, j) in zip(x, w):
            net += i * j
        return net

    def __init__(self, fun, xs):
        self.true_fun = []
        self.xs = xs
        for i in self.xs:
            x1, x2, x3, x4 = i
            self.true_fun.append(fun(x1, x2, x3, x4))

        self.sum_errors = []
        self.weights = []
        self.y_exit = []

    def go(self, mass):

        self.sum_errors = []
        self.weights = []

        w0 = w1 = w2 = w3 = w4 = 0

        self.y_exit = []
        epoch_count = 0
        while True:
            sum_error = 0
            y_ = []

            w0n = w0
            w1n = w1
            w2n = w2
            w3n = w3
            w4n = w4

            for (xi, i_fun) in zip(self.xs, range(0, 16, 1)):

                net = self.f_net(xi, [w1, w2, w3, w4], w0)
                net_y_new = self.f_net(xi, [w1n, w2n, w3n, w4n], w0n)

                y = f_y(mass[0](net))
                y_new = f_y(mass[0](net_y_new))
                y_.append(y)

                error = self.true_fun[i_fun] - y
                error_y_new = self.true_fun[i_fun] - y_new

                sum_error += abs(error)

                w0n += 0.3 * error_y_new * 1 * mass[1](net_y_new)
                w1n += 0.3 * error_y_new * xi[0] * mass[1](net_y_new)
                w2n += 0.3 * error_y_new * xi[1] * mass[1](net_y_new)
                w3n += 0.3 * error_y_new * xi[2] * mass[1](net_y_new)
                w4n += 0.3 * error_y_new * xi[3] * mass[1](net_y_new)

            self.weights.append([round(w0, 3), round(w1, 3), round(w2, 3), round(w3, 3), round(w4, 3)])
            self.y_exit.append(y_)
            self.sum_errors.append(sum_error)

            w0 = w0n
            w1 = w1n
            w2 = w2n
            w3 = w3n
            w4 = w4n

            if sum_error == 0:
                break
        return self.y_exit, self.weights, self.sum_errors

    def start(self, mass):

        y_exit, weights, sum_errors = self.go(mass)

        k = 0
        print('_' * 150)
        print('||#| iteration |#|         w0         w1         w2         w3         w4          '
              '|#|                     y values                       |#|error|#||')
        print('=' * 150)
        for i in range(0, len(y_exit)):
            print('||#|   ', "%5d" % i, ' |#| ', end='   (')
            for j in weights[i]:
                print(' ', '%-7s' % str(j), end='  ')
            print(')    |#| ', y_exit[i], ' |#| ', sum_errors[i], ' |#||')
            k += 1
        print('-' * 150)
        print_diagram(range(0, k), sum_errors)


class MinNeuralWeights:
    def __init__(self, fun, xs):
        self.fun = fun
        self.true_fun = []
        self.xs = xs
        for i in self.xs:
            x1, x2, x3, x4 = i
            self.true_fun.append(fun(x1, x2, x3, x4))

    def f_net(self, x, w, w0):
        net = w0
        for (i, j) in zip(x, w):
            net += i * j
        return net

    def check_ok(self, weights, fun):
        for (xi, i_fun) in zip(self.xs, range(0, 16, 1)):
            net = self.f_net(xi, [weights[1], weights[2], weights[3], weights[4]], weights[0])
            y = f_y(fun(net))

            if self.true_fun[i_fun] - y != 0:
                return False
        return True

    def start(self, mass):
        res = []
        for i in range(1, 16, 1):
            for j in itertools.combinations(xs, 16 - i):

                used_x = list(j)
                neural = NeuralS(self.fun, used_x)
                y_exit, weights, sum_errors = neural.go(mass)

                if self.check_ok(weights[-1], f4):
                    res.append([y_exit, weights[-1], sum_errors, used_x, len(weights)])
                    break

        print(" MIN X_RANGES LEN : ", 16 - len(res))
        print(" EPOCH COUNT : ", res[-1][4])
        for (i, j) in zip(res[-1][3], range(0, len(res[-1][3]))):
            print(" x", j, " = ", i)
        print(" w = ", res[-1][1])


if __name__ == '__main__':
    print_data()

    # 1
    print('=' * 150)
    print('|' * 150, '\n')

    n1 = NeuralS(fun, xs)
    n1.start([f1, f1_])

    print()
    print('|' * 150)
    print('=' * 150)
    print('\n')

    # 2
    print('=' * 150)
    print('|' * 150, '\n')

    n2 = NeuralS(fun, xs)
    n2.start([f4, f4_])

    print()
    print('|' * 150)
    print('=' * 150)
    print('\n')
    # 3

    print(' ' * 50)
    print('=' * 50)
    print('|' * 50, '\n')

    n3 = MinNeuralWeights(fun, xs)
    n3.start([f4, f4_])

    print()
    print('|' * 50)
    print('=' * 50)
    print(' ' * 50, '\n')
