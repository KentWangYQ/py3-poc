from matplotlib.pylab import plt
from mpl_toolkits.mplot3d import Axes3D
from .common import *


def function_2_show(x):
    fig = plt.figure()
    ax = Axes3D(fig)
    _x, _y = np.meshgrid(x[0], x[1])
    _z = function_2(np.array([_x, _y]))
    ax.plot_surface(_x, _y, _z,
                    rstride=1,
                    cstride=1,
                    cmap='rainbow'
                    )
    plt.show()


def numerical_gradient_show(x, f):
    x, y = np.meshgrid(*x)

    x = x.flatten()
    y = y.flatten()

    grad = numerical_gradient(f, np.array([x, y]))

    plt.figure()
    plt.quiver(x, y, -grad[0], -grad[1], angles='xy', color='#666666')
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()
