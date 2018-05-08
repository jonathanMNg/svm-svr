from random import uniform

# n: number of points
# x_s: lowerbound x
# x_e: upperbound x
# y_s: lowerbound y
# y_e: upperbound y
# margin: margin from divider
def make_uniform(n, x_s, x_e, y_s, y_e, margin):
    """Generate points that meet the margin requirement for the arbitrary
    and optimal separation line y = x.
    """
    dataset = []

    for __ in range(n):
        x = uniform(x_s, x_e)
        y = uniform(y_s, y_e)
        distance = abs(x-y) / 2**0.5
        while distance < margin:
            x = uniform(x_s, x_e)
            y = uniform(y_s, y_e)
            distance = abs(x-y) / 2**0.5
        dataset.append([x, y])

    labels = [1 if y > x else -1 for x, y in dataset]

    return dataset, labels
def make_poly(n, x_s, x_e, y_s, y_e, margin):
    dataset = []
    for __ in range(n):
        x = uniform(x_s, x_e)
        y = uniform(y_s, y_e)
        distance = abs(x-y) / 2**0.5
        while distance < margin:
            x = uniform(x_s, x_e)
            y = uniform(y_s, y_e)
            distance = abs(x-y) / 2**0.5
        dataset.append([x, y])
    labels = [1 if (y > 0 and x < 0 ) else -1 for x, y in dataset]

    return dataset, labels
