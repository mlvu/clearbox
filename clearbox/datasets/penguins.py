from palmerpenguins import load_penguins
import numpy as np

PENGUIN_SPECIES = ['Adélie', 'Chinstrap', 'Gentoo']

def penguins_sex(n=160):
    """
    Loads a 2d subset of the Palmer penguin data labeled with sex.

    The two features are flipper length (dm) and body mass (kg). The target class is sex. Penguins with indeterminate
    sex were filtered out.

    -- These units are chosen so that the data has roughly similar scale on both features.

    :param n: The numnber of instances to sample from the data
    :return:
    """

    penguins = load_penguins()

    penguins = penguins[penguins.sex.astype('str') != 'nan']
    penguins.sex = penguins.sex.astype('category')

    penguins.species = penguins.species.astype('category')

    penguins.flipper_length_mm /= 100
    penguins.body_mass_g /= 1000

    penguins = penguins.sample(frac=1)

    x = penguins[['flipper_length_mm', 'body_mass_g']].values

    x_train = x[:n, :]
    x_test = x[n:, :]

    y = penguins.sex.cat.codes.values  # convert the categories to integers

    y_train = y[:n]
    y_test = y[n:]

    return (x_train, y_train), (x_test, y_test)

def penguins_mass(n=160):
    """
    Loads a 1d subset of the penguin data, labeled with body mass. Used for visualizing regression.


    :param n:
    :return:
    """

    penguins = load_penguins()

    penguins = penguins[penguins.sex.astype('str') != 'nan']
    penguins.sex = penguins.sex.astype('category')

    penguins.flipper_length_mm /= 100
    penguins.body_mass_g /= 1000

    penguins = penguins.sample(frac=1)

    N = 160

    x = penguins.flipper_length_mm.values
    x_train = x[:N]
    x_test  = x[N:]

    y = penguins.body_mass_g.values
    y_train = y[:N]
    y_test = y[N:]

    return (x_train, y_train), (x_test, y_test)

def penguins_species(n=160, normalize=True):
    """
    Loads a 2d subset of the penguin data, labeled with species. Used for clustering in the examples.

    The features are bill length and bill depth in mm.

    :param n:
    :return:
    """

    # reload penguins

    penguins = load_penguins()

    penguins = penguins[penguins.sex.astype('str') != 'nan']
    penguins.sex = penguins.sex.astype('category')

    penguins.species = penguins.species.astype('category')

    penguins.flipper_length_mm /= 100
    penguins.body_mass_g /= 1000

    penguins = penguins.sample(frac=1)

    N = 160

    x = penguins[['bill_length_mm', 'bill_depth_mm']].values
    x_train = x[:N, :]
    x_test  = x[N:, :]

    y = penguins.species.cat.codes.values
    y_train = y[:N]
    y_test = y[:N]

    # Normalize the data
    if normalize:
        rng = np.ptp(x_train, axis=0, keepdims=True)
        low = x_train.min(axis=0, keepdims=True)

        x_train = (x_train - low) / rng
        x_test = (x_test - low) / rng

    return (x_train, y_train), (x_test, y_test)

