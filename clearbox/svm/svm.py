import numpy as np

class SVM():
    """
    Soft-margin support vector machine classifier.

    - Binary classification only. The 1-vs-N tactic is the most popular approach
    - The SVM problem can be solved in its primal or dual form. The search method decides which is used.
        - If the primal problem is solved, the members `w` and `b` will be set. These are the weight vector and bias (i.e.
          the coefficients and intercept) for a linear classifier. This is only possible without a kernel
        - If the dual problem is solved, the member a will be set (a vector with as many elements as there are instances
          in the data). If no kernel is used, these can be transformed to parameters `w` and `b` for a linear classifier.
    - The member `dual_used` indicates whether the dual or primal problem was solved.

    """
    def __init__(self, c=1.0, search=None, kernel=None):
        """
        :param c: The complexity parameter. The higher this value, the less likely points are to end up
            inside the margin (i.e. on the wring side of the boundary). Set to float('inf') for a hard margin classifier.
        :param solver: Algorithm used to find a solution. Available:
            - gd_search: Basic gradient descent. Does not work with a kernel.
            - projected_gd_search: Projected gradient descent on the dual problem. Simple algorithm.
            - smo_search: Sequential minimal optimization on the dual problem. Recommended.
        """
        self.c = c
        self.kernel = kernel

        self.search = gd_search if search is None else search

    def fit(self, x, labels, **kwargs):
        """
        Fit the SVM to the data. The keyword arguments are passed on to the search method, each search method has
        different hyperparameters. See the methods themselves for details.

        :param x:
        :param labels:
        :return:
        """

        self.search(self, x, labels, **kwargs)

    def compute_primal_params(self, x, labels, eps = 1e-12):
        """
        Computes the explicit parameters of the hyperplane classifier in feature space as a weight vector w and bias
        scalar b.

        :return:
        """
        assert self.solved_dual and (self.kernel is None), 'Parameters should only be computed if the dual problem was' \
                                                             ' solved and no kernel was used.'
        y = labels * 2 -1
        w = (y[:, None] * self.a[:, None] * x).sum(axis=0)

        # computation of b, see Bishop, section 7.1, page 334, eq 7.37
        xs = x @ x.T
        xs = (xs * self.a[None, :] * y[None, :]).sum(axis=1)

        # print((0 < self.a), (self.a < self.c))
        # print((0 < self.a) & (self.a < self.c))

        num_s = (self.a > eps).sum() # number of support vectors

        b = (y * xs).sum() / num_s

        return w, b

def gd_search(svm : SVM, x, labels, lr=0.001, grad_eps=1e-10, verbose=False, max_its=10_000, print_every=1_000):
    """
    Basic gradient descent algorithm. This approach works by translating the primal problem to an unconstrained one
    by introducing slack variables and substituting the constraints into the objective function.

    Simple algorithm, but does not work with kernels.

    This algorithm is included because it's easier to understand than the other search methods. However, it seems to be
    pretty sensitive to the search hyperparameters (especially for small datasets), so it's probably not the best
    approach to use in practice.

    :param x:
    :param labels:
    :return:
    """
    if svm.kernel is not None:
        raise Exception('GD solver only works with non-kernel SVM. Set kernel to None or choose a different search. ' +
                        'The SMO search is recommended')

    # TODO generic data checking in tools

    num_instances, num_features = x.shape

    # Initialize the parameters
    svm.w = np.random.randn(num_features,) # weight vector
    svm.b = np.zeros(shape=(1,)) # bias scalar

    # We are solving the _primal_ problem, not the dual
    svm.solved_dual = False

    # The y vector gives us the class labels with -1 for the negative class and +1 for the positive class.
    # since `labels` is a binary vector (with values 0 or 1), we'll transform it to the y vector.
    y = labels * 2 - 1

    # The loss function that we want to minimize is
    #    loss(w, b) =  0.5 * w^Tw + c * sum_x,y max(0, 1 - y(w^Tx + b))
    # We won't need to implement this function, because in gradient descent we only need to compute its gradient.
    #
    # With respect to one element w_i of w, the partial derivative is
    #    w_i - c * sum_(x,y in S) y*x_i
    # with S the set of x, y pairs for which y * x_i if y(w^Tx + b) >= 1 and 0 otherwise. Note that i indexes over the
    # features, not the instances.
    #
    # With respect to the bias b, we get
    #    sum_x,y = - c sum_(x, y in S) y
    # with S the same as before.

    # Here are the vectorized versions of these two functions. We'll start with the simplest, the gradient for b.
    def grad_b(w, b):
        condition = (y * (x @ w + b) >= 1) # a binary vector of size (num_instances, )

        return - svm.c * (~condition * y).sum()

    def grad_w(w, b):
        condition = (y * (x @ w + b) >= 1) # a binary vector of size (num_instances, )

        # Next, we turn `condition` into a matrix of size (num_instances, num_features) whose columns are 0 if the
        # condition `y(w^Tx + b) >= 1` is false for the instance x corresponding to that column, and x otherwise.
        xt = ~condition[:, None] * x * y[:, None]
        # -- Note the use of broadcasting: we extend `condition` to a (num_instances, 1) matrix, which is automatically
        #    extended to a (num_instances, num_features) matrix (same for y)

        return w - svm.c * xt.sum(axis=0)
        # -- We sum xt over all instances, so that we get two vectors of shape (num_features,) to sum together.

    # Now we can implement the basic gradient descent. Since we know that the problem is convex, we can search until
    # the gradient is close enough to zero.
    grad_norm = float('inf')
    iterations = 0
    while(grad_norm > grad_eps and iterations < max_its):

        # compute the gradients
        gw = grad_w(svm.w, svm.b)
        gb = grad_b(svm.w, svm.b)

        # compute the gradient norm
        grad_norm = np.sqrt(np.dot(gw, gw) + gb * gb)
        if verbose and iterations % print_every == 0:
            print(f'iteration {iterations:04}: w {svm.w}, b {svm.b}, gradient norm {grad_norm:.4} ')
            print('cond', (y * (x @ svm.w + svm.b) >= 1))

        # gradient update
        svm.w -= lr * gw
        svm.b -= lr * gb
        iterations += 1

def projected_gd_search(svm : SVM, x, labels, lr=0.001, grad_eps=1e-10, verbose=False, max_its=10_000, print_every=1_000):
    """
    A simple way to enrich gradient descent with constraints is to add a projection step. The gradient descent step is
    allowed to move around freely, but after each step, the solution is projected back into the constraint region.

    We can apply this to the dual problem of the SVM:
    ```
        minimize   sum_i a_i - sum_ij a_ia_j y_i_j x_i^Tx_j
        such that  0 <= a_i <= c for all i
        and        sum_i a_iy_i = 0
    ```
    Here, a_i are the Lagrange multipliers: we introduce one of these for each instance in our data. We optimize the
    values of the multipliers and derive the values of w and b from them if necessary.

    The gradient of the loss with respect to a_i is easy to work out. It's:
        1 - sum_j a_j y_iy_j x_i^Tx_j

    To project a given solution (a vector of all a_i's) back to the constraint region, we need to find the nearest point
    in the constraint region. For the first constraint, this is easy: we just clip the values of a_i to the range [0, c].

    The second constraint can be seen as a dot product between a vector a and a vector y. This dot product should be 0,
    or in other words, a should be projected onto the null space of y. This can be done by projecting a onto y, and then
    subtracting that vector from a: a' = a - y * (a^Ty)/(y^Ty)
    -- _This is sometimes called the vector rejection of a from y._

    We iterate this process, taking steps of gradient descent and projecting back to the constraint space.

    :param svm:
    :param x:
    :param labels:
    :return:
    """

    n, f = x.shape

    # The y vector gives us the class labels with -1 for the negative class and +1 for the positive class.
    # since `labels` is a binary vector (with values 0 or 1), we'll transform it to the y vector.
    y = labels * 2 - 1

    # We are solving the dual problem
    svm.solved_dual = True

    # We initialize the vector of a's uniformly within the first constraint.
    svm.a = np.random.random(size=(n, )) * svm.c

    # gradient of the (dual) loss wrt. a
    def grad_a(a):
        xs = x @ x.T
        ys = y[:, None] @ y[None, :]

        return (ys * xs * a[None, :]).sum(axis=1) - 1


    grad_norm = float('inf')
    iterations = 0
    while(grad_norm > grad_eps and iterations < max_its):
        ga = grad_a(svm.a)

       # compute the gradient norm
        grad_norm = np.sqrt(np.dot(svm.a, svm.a))

        # gradient update
        svm.a -= lr * ga

        # projection
        for _ in range(10):
            # -- second constraint
            svm.a = svm.a - y * np.dot(svm.a, y)/np.dot(svm.a, svm.a)
            # -- first constraint
            svm.a = np.clip(svm.a, 0, svm.c)

        if verbose and iterations % print_every == 0:
            print(f'iteration {iterations:04}: a {svm.a}, gradient norm {grad_norm:.4} ')

        iterations += 1

def smo_search(svm : SVM, x, labels):
    """
    Simple implementation of the Sequential Minimal Optimization (SMO) algorithm. This is the standard solver used
    for support vector machine. It is specific to the SVM dual problem, and does not apply to any other optimization
    problems.

    It's quick and robust enough that for simple problems, it should be usable without changing any hyperparameters.

    ## References
    - CS 229, Autumn 2009 The Simplified SMO Algorithm

    :param x:
    :param labels:
    :return:
    """

    n, f = x.shape

    # The y vector gives us the class labels with -1 for the negative class and +1 for the positive class.
    # since `labels` is a binary vector (with values 0 or 1), we'll transform it to the y vector.
    y = labels * 2 - 1

    # We are solving the dual problem
    svm.solved_dual = True

    # We initialize the vector of a's uniformly within the first constraint.
    svm.a = np.random.random(size=(n, )) * svm.c



