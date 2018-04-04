import numpy as np
import scipy.sparse as sp
from scipy.special import logsumexp, xlogy
from sklearn.cluster import KMeans


class DCSBM:
    """
    Implements a baseline Degree-corrected Stochastic Block Model fitted with variational EM.
    """

    def __init__(self, A, K, tol=1e-5, max_iter=1000, verbose=False):
        """
        :param A: scipy.sparse.spmatrix
                Sparse unweighted undirected adjacency matrix with no self-loops.
        :param K: int
                Number of clusters.
        :param tol: float
                Tolerance used for early stopping when maximizing the ELBO.
        :param max_iter: int
                Maximum number of iterations to run variational EM.
        :param verbose: bool
                Verbosity.
        """
        assert np.all(np.unique(A.data) == [1])  # unweighted
        assert A.diagonal().sum() == 0  # no self-loops
        assert (A != A.T).nnz == 0  # undirected

        self.A = A
        self.K = K
        self.N = A.shape[0]
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

        self.theta = A.sum(1).A1
        self.neighbors = A.tolil().rows

    def _init_var(self):
        self.z = self._init_spectral()  # initialize with spectral clustering using normed Laplacian
        self._m_step()  # call m_step to initialize

    def _init_spectral(self):
        Dmh = sp.diags(np.power(self.A.sum(1).A1, -1 / 2))
        L = sp.eye(self.N) - Dmh.dot(self.A).dot(Dmh)
        l, U = sp.linalg.eigsh(L, k=self.K, which='SM')

        U = U / np.linalg.norm(U, axis=1)[:, None]
        z = KMeans(self.K).fit_predict(U)
        Z = np.eye(self.K)[z]

        return Z

    def _m_step(self):
        # update eta
        self.mkk = self.z.T.dot(self.A.dot(self.z))
        self.dkg = self.theta.dot(self.z)
        self.eta = np.maximum(self.mkk / (self.dkg * self.dkg[:, None]), 1e-14)

        # update log_pi
        self.log_pi = np.log(self.z.sum(0) / self.N)

    def _update_zi(self, i):
        nbrs = self.neighbors[i]
        zi = (- self.theta[i] * (1 - self.theta[i] * self.z[i].dot(self.eta))
              - 0.5 * (self.theta[i] ** 2) * self.eta.diagonal()
              + (np.log(self.theta[nbrs])[:, None] + np.log(self.theta[i]) + self.z[nbrs].dot(np.log(self.eta))).sum(0)
              + np.log(self.theta[nbrs]).sum() + len(nbrs) * np.log(self.theta[i]) + self.z[nbrs].dot(
            np.log(self.eta)).sum(0)
              )

        zi = self.log_pi + zi
        self.z[i] = np.exp(zi - logsumexp(zi))

    def _e_step(self):
        for i in range(self.N):
            self._update_zi(i)

    def _elbo(self):
        log_p = (0.5 * (self.mkk * np.log(self.eta) - np.outer(self.dkg, self.dkg) * self.eta).sum()
                 + np.log(self.theta).dot(self.theta))
        log_p += (self.z * self.log_pi).sum()
        entropy = xlogy(self.z, self.z).sum()

        return log_p - entropy

    def fit(self):
        self._init_var()
        elbo = self._elbo()

        for it in range(self.max_iter):
            self._e_step()
            self._m_step()

            new_elbo = self._elbo()
            if np.abs(new_elbo - elbo) < self.tol:
                break
            else:
                elbo = new_elbo

            if self.verbose:
                print('it: {:3d}, elbo: {:.5f}'.format(it, elbo))

    def predict(self):
        return self.z.argmax(1)

    def fit_predict(self):
        self.fit()
        return self.predict()
