import tensorflow as tf
import numpy as np
from .dcsmb import DCSBM


class PAICAN:
    """
    Implementation of the method proposed in the paper:
    'Bayesian Robust Attributed Graph Clustering: Joint Learning of Partial Anomalies and Group Structure'
    by Aleksandar Bojchevski and Stephan Günnemann,
    published at the 32nd AAAI Conference on Artificial Intelligence (AAAI-18).

    Copyright (C) 2017
    Aleksandar Bojchevski
    Technical University of Munich
    """

    def __init__(self, A, X, K, init='dcsbm', at_once=True, tol=1e-3, max_iter=1000, verbose=False, seed=0):
        """
        :param A: scipy.sparse.spmatrix
                Sparse unweighted undirected adjacency matrix with no self-loops.
        :param X: scipy.sparse.spmatrix
                Sparse binary attribute matrix.
        :param K: int
                Number of clusters.
        :param init: string
                Initialization strategy, with 'cat': random categorical or 'dcsbm': output of baseline DCSBM.
        :param at_once: bool
                Whether to perform standard coordinate ascent or update all latent variables at once.
                If true update all z_i (resp. c_i) at once; else update each z_i (c_i) one at a time.
                Set at_once=True for great speedup and comparable performance.
        :param tol: float
                Tolerance used for early stopping when maximizing the ELBO.
        :param max_iter: int
                Maximum number of iterations to run variational EM.
        :param verbose: bool
                Verbosity.
        :param seed: int
                Seed.
        """
        assert np.all(np.unique(A.data) == [1])  # unweighted
        assert A.diagonal().sum() == 0  # no self-loops
        assert (A != A.T).nnz == 0  # undirected
        assert np.all(np.unique(X.data) == [1])  # unweighted
        assert (A.sum(1).A1 == 0).sum() == 0 # no singletons

        tf.reset_default_graph()

        # general params
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.at_once = at_once

        # shortcut aliases
        sp_dot = tf.sparse_tensor_dense_matmul
        dot = tf.matmul
        eps = 1e-20
        # add a small positive constant to the log to avoid numerical instability
        safe_log = lambda x: tf.log(x + 1e-7)

        # input data and placeholders
        self.idx = tf.placeholder(tf.int32, shape=[None])
        self.X = tf.SparseTensor(*self.__sparse_feeder(X.astype(np.float32)))
        self.Xm = tf.constant(1 - X.toarray(), tf.float32, name='Xm')
        self.A = tf.SparseTensor(*self.__sparse_feeder(A.astype(np.float32)))
        self.K = K

        self.N, D = X.shape

        # priors
        alpha = 10 * tf.ones([self.K])
        beta = tf.constant(np.array([10, 1, 1, 1]), tf.float32)

        tf.set_random_seed(seed)

        if init == 'cat':
            # initialize the cluster assignment randomly
            z_init = tf.contrib.distributions.Categorical(tf.ones([self.K]) / tf.cast(self.K, tf.float32)).sample(self.N)
            z_init = tf.gather(tf.eye(self.K), z_init) + eps
            z_init = z_init / tf.reduce_sum(z_init, 1)[:, None]
        elif init == 'dcsbm':
            dcsbm = DCSBM(A, K)
            dcsbm.fit()
            z_init = dcsbm.z
        else:
            raise ValueError
        
        self.z = tf.Variable(z_init, name='z', dtype=tf.float32)

        # initialize the anomaly indicators
        c_init = tf.reshape(tf.tile([0.997, 0.001, 0.001, 0.001], [self.N]), [self.N, 4])
        self.c = tf.Variable(c_init, name='c')

        # materialize the partial graph/attribute anomaly views
        self.ca = tf.stack([self.c[:, 0] + self.c[:, 2], self.c[:, 1] + self.c[:, 3]], 1)
        self.cx = tf.stack([self.c[:, 0] + self.c[:, 1], self.c[:, 2] + self.c[:, 3]], 1)

        # MLE update of the degree parameters
        theta = tf.maximum(sp_dot(self.A, self.ca[:, 0][:, None]), eps)
        theta_p = tf.sparse_reduce_sum(self.A, 0) + eps

        # MLE update the mkk/eta parameters
        zc = self.ca[:, 0][:, None] * self.z
        mkk = dot(tf.transpose(zc), sp_dot(self.A, zc))
        mbg = dot(self.ca[:, 1][None, :], sp_dot(self.A, self.ca[:, 0][:, None]))
        mbb = dot(self.ca[:, 1][None, :], sp_dot(self.A, self.ca[:, 1][:, None]))

        dkg = tf.transpose(dot(tf.transpose(self.ca[:, 0][:, None] * theta), self.z))
        DB = dot(tf.transpose(theta_p[:, None]), self.ca[:, 1][:, None])
        g = tf.reduce_sum(self.ca[:, 0])

        eta = tf.maximum(mkk / (dkg * tf.transpose(dkg)), eps)
        etabg = tf.squeeze(tf.maximum(mbg / (DB * g), eps))
        etabb = tf.squeeze(tf.maximum(mbb / (DB ** 2), eps))

        thetap_B = tf.squeeze(dot(tf.transpose(theta_p[:, None]), self.ca[:, 1][:, None]))

        # MLE update of the topics
        rx = self.cx[:, 0][:, None] * self.z
        topics = tf.clip_by_value(sp_dot(tf.sparse_transpose(self.X), rx) / tf.reduce_sum(rx, 0), eps, 1 - eps)

        # MAP update of the cluster/anomaly priors
        log_pi = safe_log((tf.reduce_sum((1 - self.c[:, 3][:, None]) * self.z, 0) + alpha) /
                        (tf.reduce_sum(1 - self.c[:, 3])) + tf.reduce_sum(alpha))
        log_rho = safe_log((tf.reduce_sum(self.c, 0) + beta) / (tf.cast(self.N, tf.float32) + tf.reduce_sum(beta)))

        # variational update of the cluster assignments z
        z_sbm = self.ca[:, 0][:, None] * (
            -theta * (1 - theta * self.ca[:, 0][:, None] * dot(self.z, eta))
            - 0.5 * theta ** 2 * tf.diag_part(eta)
            + dot(sp_dot(self.A, zc), safe_log(eta))
            + sp_dot(self.A, (self.ca[:, 0][:, None] * safe_log(theta)))
            + sp_dot(self.A, self.ca[:, 0][:, None]) * safe_log(theta))

        x_log_topics = sp_dot(self.X, safe_log(topics)) + dot(self.Xm, safe_log(1 - topics))
        z_att = self.cx[:, 0][:, None] * x_log_topics
        z_prior = (1 - self.c[:, 3])[:, None] * log_pi

        z_update = z_sbm + z_att + z_prior
        z_update_norm = tf.exp(z_update - tf.reduce_logsumexp(z_update, 1)[:, None])
        self.op_update_z = tf.assign(self.z, z_update_norm)
        self.op_update_zi = tf.scatter_update(self.z, self.idx, tf.gather(z_update_norm, self.idx))

        # variational update of the anomaly assignments c
        good_sbm = (
            -theta
            + theta ** 2 * tf.diag_part(dot(dot(self.z, eta), tf.transpose(self.z)))[:, None] * self.ca[:, 0][:, None]
            - etabg * (thetap_B - self.ca[:, 1][:, None] * theta_p[:, None])
            - 0.5 * theta ** 2 * dot(self.z, tf.diag_part(eta)[:, None])
            + sp_dot(self.A * dot(dot(self.z, safe_log(eta)), tf.transpose(self.z)), self.ca[:, 0][:, None])
            + sp_dot(self.A, self.ca[:, 0][:, None] * safe_log(theta))
            + sp_dot(self.A, self.ca[:, 0][:, None]) * safe_log(theta)
            + sp_dot(self.A, (safe_log(theta_p * etabg) * self.ca[:, 1])[:, None]))

        corr_sbm = (
            - theta_p[:, None] * etabb * (thetap_B - theta_p[:, None] * self.ca[:, 1][:, None])
            - etabg * theta_p[:, None] * (g - self.ca[:, 0][:, None])
            - 0.5 * theta_p[:, None] ** 2 * etabb
            + sp_dot(self.A, self.ca[:, 1][:, None] * safe_log(theta_p * etabb)[:, None])
            + sp_dot(self.A, self.ca[:, 1][:, None]) * safe_log(theta_p[:, None])
            + sp_dot(self.A, self.ca[:, 0][:, None]) * safe_log(theta_p[:, None] * etabg))

        good_att = tf.reduce_sum(self.z * x_log_topics, 1)
        corr_att = tf.cast(D, tf.float32) * safe_log(0.5)

        c_update = tf.squeeze(tf.stack([
            good_sbm + good_att[:, None] + log_rho[0],
            corr_sbm + good_att[:, None] + log_rho[1],
            good_sbm + corr_att + log_rho[2],
            corr_sbm + corr_att + log_rho[3] - dot(self.z, log_pi[:, None])
        ], 1))

        c_update_norm = tf.exp(c_update - tf.reduce_logsumexp(c_update, 1)[:, None])

        self.op_update_c = tf.assign(self.c, c_update_norm)
        self.op_update_ci = tf.scatter_update(self.c, self.idx, tf.gather(c_update_norm, self.idx))

        # ELBO
        logp_sbm = tf.squeeze(
            0.5 * tf.reduce_sum(mkk * safe_log(eta) - (dkg * tf.transpose(dkg)) * eta)
            + dot(tf.transpose(safe_log(theta)), theta * self.ca[:, 0][:, None])
            + mbg * safe_log(etabg)
            + 0.5 * mbb * safe_log(etabb)
            - 0.5 * DB * DB * etabb
            - DB * g * etabg
            + dot(tf.transpose(safe_log(theta_p[:, None])), theta_p[:, None] * self.ca[:, 1][:, None])
        )

        prior = tf.reduce_sum(dot(tf.transpose(self.z), 1 - self.c[:, 3][:, None]) * log_pi) \
                + tf.reduce_sum(dot(self.c, log_rho[:, None]))

        logp_att = tf.reduce_sum(self.cx[:, 0][:, None] * self.z * x_log_topics) \
                   + tf.reduce_sum(self.cx[:, 1]) * tf.cast(D, tf.float32) * safe_log(0.5)

        entropy = tf.reduce_sum(self.c * safe_log(self.c)) + tf.reduce_sum(self.z * safe_log(self.z))

        logp = logp_sbm + logp_att + prior
        ELBO = logp - entropy

        self.ELBO = ELBO

    def __sparse_feeder(self, M):
        """
        Transforms the sparse matrix M into a format suitable for Tensorflow's sparse placeholder.

        :param M: scipy.sparse.spmatrix
                Sparse matrix.
        :return: Indices of non-zero elements, their values, and the shape of the matrix
        """
        M = M.tocoo()
        return np.vstack((M.row, M.col)).T, M.data, M.shape

    def _em(self, sess):
        if self.at_once:
            sess.run(self.op_update_z)
            sess.run(self.op_update_c)
        else:
            for i in range(self.N):
                sess.run(self.op_update_zi, {self.idx: [i]})
            for i in range(self.N):
                sess.run(self.op_update_ci, {self.idx: [i]})

    def fit_predict(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            prev_elbo = -np.inf

            for it in range(self.max_iter):
                elbo = self.ELBO.eval()
                if self.verbose:
                    print('iter {:3d}, ELBO: {:.5f}'.format(it, elbo))

                # terminating conditions
                if np.abs(prev_elbo - elbo) < self.tol:
                    break
                else:
                    prev_elbo = elbo
                    self._em(sess)

            pr_z = self.z.eval().argmax(1)
            pr_ca = self.ca.eval().argmax(1)
            pr_cx = self.cx.eval().argmax(1)

            return pr_z, pr_ca, pr_cx