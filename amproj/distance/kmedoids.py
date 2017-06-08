"""Contains implementations for k-medoids algorithm"""

import datetime
import random
try:
    import numpy as np
except ImportError:
    pass

from operator import mul


class FuzzyKMedoids:

    def __init__(self,
                 n_clusters=7,
                 m=1.6,
                 s=1,
                 n_iterations=10000,
                 epsilon=0.01,
                 q=3):
        """Initializes a new instance of the FuzzyKMedoids partitioner.

        Parameters
        ----------
        n_clusters : int
            The number of clusters that should form this partition.
        m : float
            A float parameter that controls the fuzziness of membership for
            each object. Must be set to a value between 1.0 and infinity.
        s : float
            The exponent of the relevance weights for the dissimilarity
            matrices. Its value must be set between 1 and infinity.
        n_iterations : int
            The maximum number of iterations before the algorithm returns
        epsilon : float
            The minimum change in the adequacy criterion that will cause
            the algorithm to continue for another iteration.
        q : int
            The number of elements in each cluster's prototype. A cluster's
            prototype is a list of elements that represent that cluster.
        """
        if type(n_clusters) != int or n_clusters < 2:
            raise ValueError(
                "The number of clusters must be an integer higher than 1")
        if (type(m) != float and type(m) != int) or m <= 1.0:
            raise ValueError(
                "The parameter 'm' must be a number higher than 1")
        if (type(s) != float and type(s) != int) or s < 1:
            raise ValueError(
                "The parameter 's' must be a number and at least 1")
        if type(n_iterations) != int or n_iterations < 1:
            raise ValueError(
                "The number of iterations must be a positive integer")
        if type(epsilon) != float:
            raise ValueError("The minimum difference that will cause " +
                             "cause iteration must be a float")
        if type(q) != int or q < 1:
            raise ValueError(
                "The parameter 'q' must be a positive integer")
        self.n_clusters = n_clusters
        self.m = m
        self.s = s
        self.n_iterations = n_iterations
        self.epsilon = epsilon
        self.q = q

    def __membership_degress__(self,
                               views,
                               K,
                               m,
                               s,
                               q,
                               p,
                               n,
                               lambs,
                               G):
        """Calculates the membership degrees for each object.

        Parameters
        ----------
        views : list<dist_matrix>
            The dissimilarity matrices corresponding to the views.
        K : int
            The number of clusters.
        m : float
            A float parameter that controls the fuzziness of membership for
            each object. Must be set to a value between 1.0 and infinity.
        s : float
            The exponent of the relevance weights for the dissimilarity
            matrices. Its value must be set between 1 and infinity.
        q : int
            The number of elements in each cluster's prototype. A cluster's
            prototype is a list of elements that represent that cluster.
        p : int
            The number of views.
        n : int
            The number of objects in the dataset.
        lambs : matrix<float>
            A matrix containing the view relevance weights per cluster. So this
            matrix has shape: K lines and p columns.
        G : matrix<int>
            The list of prototypes for each cluster as a matrix. So this matrix
            has shape: K lines and q columns. And the G[k][q] is the q-th
            prototype of the k-th cluster.

        Remarks
        -------
        This method doesn't completly check its parameters. The list is quite
        big just for making tests easier.

        Returns
        -------
        u : matrix<float>
            A membership degree matrix according to the given parameters. The
            shape of this matrix is: n lines and K columns. Each value in the
            matrix must be between 0 and 1. And the sum of the values in the
            same line must be equal to 1.
        """
        if type(K) != int:
            raise ValueError("The parameter 'K' must be an integer")
        if type(m) != float and type(m) != int:
            raise ValueError("The parameter 'm' must be a number")
        if type(s) != float and type(m) != int:
            raise ValueError("The parameter 'm' must be a number")
        if type(q) != int:
            raise ValueError("The parameter 'q' must be an integer")
        if type(p) != int:
            raise ValueError("The parameter 'p' must be an integer")
        if type(n) != int:
            raise ValueError("The parameter 'n' must be an integer")
        if len(views) != p:
            raise ValueError("The size of the views list must be p")
        if len(lambs) != K:
            raise ValueError("The parameter 'lambs' must have K lines")
        if len(lambs[0]) != p:
            raise ValueError("The parameter 'lambs' must have p columns")
        if len(G) != K:
            raise ValueError("The parameter 'G' must have K lines")
        if len(G[0]) != q:
            raise ValueError("The parameter 'G' must have q columns")
        u = []  # the membership degrees; n lines and K columns
        for i in range(n):  # initialize u to all zeros
            u.append([0.0] * K)
        for i in range(n):
            for k in range(K):
                sh = 0
                for h in range(K):
                    sjn = 0.0  # \sigma_{j=1}^{p} numerator
                    for j in range(p):
                        se = 0.0
                        for e in G[k]:
                            se += views[j][i][e]
                        sjn += (lambs[k][j] ** s) * se
                    sjd = 0.0  # \sigma_{j=1}^{p} denominator
                    for j in range(p):
                        se = 0.0
                        for e in G[h]:
                            se += views[j][i][e]
                        sjd += (lambs[h][j] ** s) * se
                    sh += (sjn / sjd) ** (1/(m - 1))
                u[i][k] = sh ** -1
        return u

    def __cost_function__(self,
                          views,
                          K,
                          m,
                          s,
                          q,
                          p,
                          n,
                          u,
                          lambs,
                          G,):
        """Calculates the cost function value for the given state.
        Parameters
        ----------
        views : list<dist_matrix>
            The dissimilarity matrices corresponding to the views.
        K : int
            The number of clusters.
        m : float
            A float parameter that controls the fuzziness of membership for
            each object. Must be set to a value between 1.0 and infinity.
        s : float
            The exponent of the relevance weights for the dissimilarity
            matrices. Its value must be set between 1 and infinity.
        q : int
            The number of elements in each cluster's prototype. A cluster's
            prototype is a list of elements that represent that cluster.
        p : int
            The number of views.
        n : int
            The number of objects in the dataset.
        u : matrix<float>
            The membership degrees of each object for each cluster. So this
            matrix must be n x K.
        lambs : matrix<float>
            A matrix containing the view relevance weights per cluster. So this
            matrix has shape: K lines and p columns.
        G : matrix<int>
            The list of prototypes for each cluster as a matrix. So this matrix
            has shape: K lines and q columns. And the G[k][q] is the q-th
            prototype of the k-th cluster.

        Remarks
        -------
        This method doesn't completly check its parameters. The list is quite
        big just for making tests easier.

        Returns
        -------
        J : float
            The adequacy criterion.
        """
        if type(K) != int:
            raise ValueError("The parameter 'K' must be an integer")
        if type(m) != float and type(m) != int:
            raise ValueError("The parameter 'm' must be a number")
        if type(s) != float and type(m) != int:
            raise ValueError("The parameter 'm' must be a number")
        if type(q) != int:
            raise ValueError("The parameter 'q' must be an integer")
        if type(p) != int:
            raise ValueError("The parameter 'p' must be an integer")
        if type(n) != int:
            raise ValueError("The parameter 'n' must be an integer")
        if len(views) != p:
            raise ValueError("The size of the views list must be p")
        if len(u) != n:
            raise ValueError("The matrix 'u' must have n lines")
        if len(u[0]) != K:
            raise ValueError("The matrix 'u' must have K columns")
        if len(lambs) != K:
            raise ValueError("The parameter 'lambs' must have K lines")
        if len(lambs[0]) != p:
            raise ValueError("The parameter 'lambs' must have p columns")
        if len(G) != K:
            raise ValueError("The parameter 'G' must have K lines")
        if len(G[0]) != q:
            raise ValueError("The parameter 'G' must have q columns")
        J = 0.0
        for k in range(K):
            for i in range(n):
                uik = u[i][k] ** m
                sj = 0.0
                for j in range(p):
                    se = 0.0
                    for e in G[k]:
                        se += views[j][i][e]
                    sj += (lambs[k][j] ** s) * se
                J += uik * sj
        return J

    def __update_prototypes__(self,
                              views,
                              K,
                              m,
                              s,
                              q,
                              p,
                              n,
                              u,
                              lambs,):
        """With the membership degree matrix, u, and the view relevance matrix,
        lambs, fixed, calculate the best possible prototypes for G that
        minimizes the adequacy criterion J.

        Parameters
        ----------
        views : list<dist_matrix>
            The dissimilarity matrices corresponding to the views.
        K : int
            The number of clusters.
        m : float
            A float parameter that controls the fuzziness of membership for
            each object. Must be set to a value between 1.0 and infinity.
        s : float
            The exponent of the relevance weights for the dissimilarity
            matrices. Its value must be set between 1 and infinity.
        q : int
            The number of elements in each cluster's prototype. A cluster's
            prototype is a list of elements that represent that cluster.
        p : int
            The number of views.
        n : int
            The number of objects in the dataset.
        u : matrix<float>
            The membership degrees of each object for each cluster. So this
            matrix must be n x K.
        lambs : matrix<float>
            A matrix containing the view relevance weights per cluster. So this
            matrix has shape: K lines and p columns.

        Returns
        -------
        G : matrix<int>
            The list of prototypes for each cluster as a matrix. So this matrix
            has shape: K lines and q columns. And the G[k][q] is the q-th
            prototype of the k-th cluster. This returned G is the one that
            minimizes the adequacy criterion given that the u and lambs
            matrices are fixed.
        """
        new_G = []  # K lines and q columns
        for k in range(K):
            G_star = []  # q elements
            uk = [0.0] * n
            if s != 1.0:
                lambsk = list(map(lambs[k], lambda x: x ** s))
            else:
                lambsk = lambs[k]
            while len(G_star) < q:
                min_e = None
                min_e_val = float("inf")  # infinity in Python 2.x
                for e_h in range(n):
                    if e_h in G_star:
                        continue
                    sum_n = 0.0
                    for i in range(n):
                        if uk[i] <= 0.0:
                            uk[i] = u[i][k] ** m
                        sum_p = 0.0
                        for j in range(p):
                            sum_p += lambsk[j] * views[j][i][e_h]
                        sum_n += uk[i] * sum_p
                    if sum_n < min_e_val:
                        min_e = e_h
                        min_e_val = sum_n
                G_star += [min_e]
            new_G += [G_star]
        return new_G

    def __update_lambs__(self,
                         views,
                         K,
                         m,
                         s,
                         q,
                         p,
                         n,
                         u,
                         G,):
        """Updates the view relevance weights given that the cluster prototypes
        and membership degree matrix, u, are fixed.

        Parameters
        ----------
        views : list<dist_matrix>
            The dissimilarity matrices corresponding to the views.
        K : int
            The number of clusters.
        m : float
            A float parameter that controls the fuzziness of membership for
            each object. Must be set to a value between 1.0 and infinity.
        s : float
            The exponent of the relevance weights for the dissimilarity
            matrices. Its value must be set between 1 and infinity.
        q : int
            The number of elements in each cluster's prototype. A cluster's
            prototype is a list of elements that represent that cluster.
        p : int
            The number of views.
        n : int
            The number of objects in the dataset.
        u : matrix<float>
            The membership degrees of each object for each cluster. So this
            matrix must be n x K.
        G : matrix<int>
            The list of prototypes for each cluster as a matrix. So this matrix
            has shape: K lines and q columns. And the G[k][q] is the q-th
            prototype of the k-th cluster.

        Returns
        -------
        lambs : matrix<float>
            A new matrix containing the view relevance weights per cluster. So
            this matrix has shape: K lines and p columns.
        """
        new_lambs = []  # should have K lines and p columns
        for k in range(K):
            new_lambs.append([0.0] * p)
            sums = [0.0] * p
            for h in range(p):
                for i in range(n):
                    uik = u[i][k] ** m
                    sum_e = 0.0
                    for e in G[k]:
                        sum_e += views[h][i][e]
                    sums[h] += uik * sum_e
            oneoverp = 1/float(p)
            for j in range(p):
                denominator = sums[j]
                numerator = reduce(mul, sums, 1) ** oneoverp  # product of sums
                new_lambs[k][j] = numerator/denominator
        return new_lambs

    def fit(self, *views, **kwargs):
        """Trains the classifier in the provided training data.

        Parameters
        ----------
        views : list<dist_matrix>
            The dissimilarity matrices corresponding to the views.
        kwargs : dict
            Some optional parameters. You can pass some function as an optional
            named parameter called "updated" and it will be executed at each
            iteration and will receive the adequacy criterion before and after
            the iteration.
        """
        if len(views) < 0:
            raise ValueError(
                "There must be at least one view to train the algorithm")
        p = len(views)
        n = len(views[0])
        for view in views:
            if len(view) != n:
                raise ValueError("All views must have the same shape")
            if len(view[0]) != n:
                raise ValueError("All views must have the same shape")
        K = self.n_clusters
        m = self.m
        s = self.s
        T = self.n_iterations
        q = self.q
        t = 0
        # initialize the "view relevance weights per cluster" matrix
        lambs = []  # should have K lines and p columns
        for k in range(K):
            if s == 1.0:  # if this is MFCMdd-RWL-P
                row = [1.0] * p
            else:
                row = [1.0/p] * p
            lambs.append(row)
        # initialize cluster prototypes
        pool = range(n)  # we need to shuffle the pool of objects
        try:
            random.shuffle(pool, np.random.uniform)
        except NameError:
            random.shuffle(pool)
        G = []  # K lines and q columns
        i = 0  # index of the next element we should take for the prototype
        for k in range(K):
            # get the next q elements for the k-th prototype
            G.append(pool[i:i+q])
            i += q
        # compute the membership degree; u has n lines and K columns
        u = self.__membership_degress__(views, K, m, s, q, p, n, lambs, G)
        # compute the cost function's value for this initial state; J is float
        J = self.__cost_function__(views, K, m, s, q, p, n, u, lambs, G)
        delta = float("inf")  # infinity
        while t < T and delta > self.epsilon:
            t += 1
            # with u and lambs fixed, find the best prototypes
            start = datetime.datetime.now()
            G = self.__update_prototypes__(views, K, m, s, q, p, n, u, lambs)
            end = datetime.datetime.now()
            print("Update prototypes in " + str(end - start))
            start = datetime.datetime.now()
            lambs = self.__update_lambs__(views, K, m, s, q, p, n, u, G)
            end = datetime.datetime.now()
            print("Update lambdas in " + str(end - start))
            delta = J
            start = datetime.datetime.now()
            u = self.__membership_degress__(views, K, m, s, q, p, n, lambs, G)
            end = datetime.datetime.now()
            print("Update membership degrees in " + str(end - start))
            start = datetime.datetime.now()
            J = self.__cost_function__(views, K, m, s, q, p, n, u, lambs, G)
            end = datetime.datetime.now()
            print("Update cost function in " + str(end - start))
            # for debug
            if "updated" in kwargs:
                kwargs["updated"](delta, J)
            delta -= J
        return lambs, G, u, J
