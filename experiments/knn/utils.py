import numpy

def inverse_squared(distances):
    if 0. in distances:
        return distances == 0

    return 1 / numpy.square(distances)