import numpy as np
import matplotlib.pyplot as plt


# Read dataset
x, y = [], []
for sample in open("../_Data/prices.txt", "r"):
    xx, yy = sample.split(",")
    x.append(float(xx))
    y.append(float(yy))
x, y = np.array(x), np.array(y)
# Perform normalization
x = (x - x.mean()) / x.std()
# Scatter dataset
plt.figure()
plt.scatter(x, y, c="g", s=20)
plt.show()

x0 = np.linspace(-2, 4, 100)


# Get regression model under LSE criterion with degree 'deg'
def get_model(deg):
    return lambda input_x=x0: np.polyval(np.polyfit(x, y, deg), input_x)


# Get the cost of regression model above under given x, y
def get_cost(deg, input_x, input_y):
    return 0.5 * ((get_model(deg)(input_x) - input_y) ** 2).sum()

# Set degrees
test_set = (1, 4, 10)
for d in test_set:
    print(get_cost(d, x, y))

# Visualize results
plt.scatter(x, y, c="g", s=20)
for d in test_set:
    plt.plot(x0, get_model(d)(), label="degree = {}".format(d))
plt.xlim(-2, 4)
plt.ylim(1e5, 8e5)
plt.legend()
plt.show()
