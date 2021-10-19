import numpy as np
import copy

from .passive import PassiveSampler
from .stratification import (Strata, stratify_by_features, stratify_by_scores,
                             auto_stratify)
from .input_verification import (verify_unit_interval, \
                                 verify_positive, \
                                 verify_scores, verify_consistency, \
                                 verify_boolean, verify_strata, \
                                 scores_to_probs)

from .oasis import BetaBernoulliModel


class StratifiedSampler(PassiveSampler):
    def __init__(self, alpha, predictions, scores, oracle, proba=False,
                 epsilon=1e-3, opt_class=None, prior_strength=None,
                 decaying_prior=True, strata=None, record_inst_hist=False,
                 max_iter=None, identifiers=None, debug=False, **kwargs):
        super(StratifiedSampler, self).__init__(alpha, predictions, oracle,
                                           max_iter, identifiers, True, debug)
        self.scores = verify_scores(scores)
        self.proba, self.opt_class = \
            verify_consistency(self.predictions, self.scores, proba, opt_class)
        self.epsilon = verify_unit_interval(float(epsilon))
        self.strata = verify_strata(strata)
        self.record_inst_hist = verify_boolean(record_inst_hist)
        self.decaying_prior = verify_boolean(decaying_prior)

        # store the estimated variance for estimator
        self._estimate_std = np.tile(np.nan, [self._max_iter, self._n_class])
        # Need to transform scores to the [0,1] interval (to use as proxy for
        # probabilities)
        self._probs = scores_to_probs(self.scores, self.proba)

        # Average the probabilities over opt_class
        self._probs_avg_opt_class = np.mean(self._probs[:,self.opt_class], \
                                            axis=1, keepdims=True)


        # Generate strata if not given
        if self.strata is None:
            if np.sum(self.opt_class) > 1:
                # If optimising over multiple classifiers, use the averaged
                # probabilities to stratify
                self.strata = \
                    auto_stratify(self._probs_avg_opt_class.ravel(), **kwargs)
            else:
                # Otherwise use scores from single classifier to stratify
                self.strata = \
                    auto_stratify(self.scores[:,self.opt_class].ravel(), \
                                  **kwargs)

        self.pos_strata_idx, self.neg_strata_idx, self.mix_strata_idx = \
            self.strata.split_by_prediction(self.predictions)

        # count positive pairs
        self.P = np.sum(self.predictions == 1)

        # Calculate mean prediction per stratum
        self._preds_avg_in_strata = self.strata.intra_mean(self.predictions)

        # Choose prior strength if not given
        if prior_strength is None:
            self.prior_strength = 2*self.strata.n_strata_
        else:
            self.prior_strength = float(prior_strength)
        # record the history of sample selection
        self.history = np.zeros((max_iter, self.strata.n_strata_), dtype=int)
        self.n_strata_ = self.strata.n_strata_
        # Instantiate Beta-Bernoulli model using probabilities averaged over
        # opt_class
        theta_0 = self.strata.intra_mean(self._probs_avg_opt_class)
        gamma = self._calc_BB_prior(theta_0.ravel())
        # Instantiate Beta-Bernoulli model using uniform prior
        # gamma = np.zeros((2,self.strata.n_strata_))
        self._BB_model = BetaBernoulliModel(gamma[0], gamma[1],
                                            decaying_prior=self.decaying_prior)
        # ground truth labels for oracle test purpose
        if "labels" in kwargs:
            self.labels = kwargs["labels"]
            self.strata_var = self.strata.intra_var(self.labels)
            self.strata_mean = self.strata.intra_mean(self.labels)
            self.TP = np.sum((self.predictions.reshape(-1) == 1) & (self.labels == 1))
            self.FP = self.P - self.TP
            self.FN = np.sum((self.predictions.reshape(-1) == 0) & (self.labels == 1))
            self.F_measure = self._F_measure(self.alpha, self.TP, self.FP, self.FN)
            partial_tp = (self.P*alpha + (1-alpha)*self.FN)/(self.P*alpha+(self.TP+self.FN)*(1-alpha))**2
            partial_fn = -(1-alpha)*self.TP/(self.P*alpha+(self.TP+self.FN)*(1-alpha))**2
            self.partial_weight = np.zeros(self.strata.n_strata_)
            # add normalization
            self.partial_weight[self.pos_strata_idx] = np.abs(partial_tp)/(np.abs(partial_tp)+np.abs(partial_fn))
            self.partial_weight[self.neg_strata_idx] = np.abs(partial_fn)/(np.abs(partial_tp)+np.abs(partial_fn))

    @property
    def estimate_std(self):
        if self.t_ == 0:
            return None
        if self._multiple_class:
            return self._estimate_std[0:self.t_,:]
        else:
            return self._estimate_std[0:self.t_,:].ravel()
    @estimate_std.setter
    def estimate_std(self, value):
        raise AttributeError("can't set attribute.")
    @estimate_std.deleter
    def estimate_std(self):
        raise AttributeError("can't delete attribute.")


    def select_stratum(self, sample_strategy="FILA-thompson", **kwargs):
        """
        Pick the next stratum to sample from according to sample_strategy
        :param sample_strategy: "ass"
        :return: stratum_idx: index of stratum selected
        """
        if sample_strategy == "FILA-thompson":
            assert len(self.mix_strata_idx) == 0
            a,b = self._BB_model.get_beta_parameter()
            mu_hat = np.random.beta(a,b)
            emp_strata_var = mu_hat * (1-mu_hat)
            emp_strata_std = np.sqrt(emp_strata_var)
            tp_est = (mu_hat * self.strata.sizes_)[self.pos_strata_idx].sum()
            fn_est = (mu_hat*self.strata.sizes_)[self.neg_strata_idx].sum()
            partial_tp = self.P*self.alpha + (1-self.alpha)*fn_est
            partial_fn = (1-self.alpha)*tp_est
            # heuristic: stop sampling from a strata if the sample size already
            # exceed K times of the population size of that strata. This is mainly
            # to avoid long execution time caused by redundant sampling
            n_sampled = self.strata._n_sampled
            strata_size = self.strata.sizes_
            K = 10 # parameter that can be adjusted
            continue_sample = (n_sampled < strata_size * K).astype(int)
            # check if each strata is fully sampled already
            # strata_empty = np.ones(self.strata.n_strata_)
            # for stratum_idx in np.arange(self.strata.n_strata_):
            #     if np.all(self.strata._sampled[stratum_idx]):
            #         strata_empty[stratum_idx] = 0
            # opt_strata_weight = emp_strata_std * self.strata.sizes_ * strata_empty
            opt_strata_weight = emp_strata_std * self.strata.sizes_ * continue_sample
            opt_strata_weight[self.pos_strata_idx] = opt_strata_weight[self.pos_strata_idx]*partial_tp
            opt_strata_weight[self.neg_strata_idx] = opt_strata_weight[self.neg_strata_idx]*partial_fn
            weight = opt_strata_weight / self.strata._n_sampled
            stratum_idx = np.argmax(weight)

        elif sample_strategy == "prop":
            stratum_idx = self.strata._sample_stratum(pmf=self.strata.weights_)

        elif sample_strategy == "neyman-thompson":
            a,b = self._BB_model.get_beta_parameter()
            mu_hat = np.random.beta(a,b)
            emp_strata_var = mu_hat * (1-mu_hat)
            emp_strata_std = np.sqrt(emp_strata_var)
            # heuristic: stop sampling from a strata if the sample size already
            # exceed K times of the population size of that strata. This is mainly
            # to avoid long execution time caused by redundant sampling
            n_sampled = self.strata._n_sampled
            strata_size = self.strata.sizes_
            K = 10 # parameter that can be adjusted
            continue_sample = (n_sampled < strata_size * K).astype(int)
            opt_strata_weight = emp_strata_std * self.strata.sizes_ * continue_sample
            weight = opt_strata_weight / self.strata._n_sampled
            stratum_idx = np.argmax(weight)

        else:
            raise Exception("sample strategy %s not implemented" % sample_strategy)

        return stratum_idx

    def _sample_item(self, **kwargs):
        """Sample an item from the pool
        """
        if 'fixed_stratum' in kwargs:
            stratum_idx = kwargs['fixed_stratum']
        else:
            # for each stratified sampling method, we first guarantee at least two points per strata
            if np.min(self.strata._n_sampled) < 2:
                stratum_idx = np.argmin(self.strata._n_sampled)
            else:
                stratum_idx = self.select_stratum(**kwargs)

        loc = self.strata._sample_in_stratum(stratum_idx,
                                                 replace=self.replace)

        return loc, 1, {'stratum': stratum_idx}


    def _update_estimate_and_sampler(self, ell, ell_hat, weight, extra_info,
                                     **kwargs):
        # updating the BB model
        self._BB_model.update(ell, extra_info['stratum'])
        #  no prior information
        a,b = self._BB_model.get_counts()
        #c = np.divide(a,a+b, out=np.zeros(a.shape, dtype=float), where=(a+b!=0))
        with np.errstate(divide='ignore', invalid='ignore'):
            c = a/(a+b)
        self._TP = (c*self.strata.sizes_)[self.pos_strata_idx].sum()
        self._FN = (c*self.strata.sizes_)[self.neg_strata_idx].sum()
        # self._TP = (a/(a+b)*self.strata.sizes_)[self.pos_strata_idx].sum()
        # self._FN = (a/(a+b)*self.strata.sizes_)[self.neg_strata_idx].sum()
        self._FP = self.P - self._TP
        # bayesian methods: use prior information
        # self._TP = (self._BB_model.theta_*self.strata.sizes_)[self.pos_strata_idx].sum()
        # self._FN = (self._BB_model.theta_*self.strata.sizes_)[self.neg_strata_idx].sum()
        # self._FP = self.P - self._TP
        self._estimate[self.t_] = \
            self._F_measure(self.alpha, self._TP, self._FP, self._FN)

        strata_var = self._BB_model.theta_ * (1-self._BB_model.theta_)
        partial_tp = (self.P *self.alpha + (1-self.alpha)*self._FN)/(self.P*self.alpha+(self._TP+self._FN)*(1-self.alpha))**2
        partial_fn = (1-self.alpha)*self._TP/(self.P*self.alpha+(self._TP+self._FN)*(1-self.alpha))**2
        with np.errstate(divide='ignore',invalid='ignore'):
            # when there is no sample from a strata. The variance will gives nan or inf. But this doesn't matter since
            # we'll soon have sample from all strata.
            var_p = ((self.strata.sizes_*partial_tp)**2*strata_var / self.strata._n_sampled)[self.pos_strata_idx].sum()
            var_n = ((self.strata.sizes_*partial_fn)**2*strata_var / self.strata._n_sampled)[self.neg_strata_idx].sum()
        est_std = np.sqrt(var_p + var_n)
        self._estimate_std[self.t_] = est_std
        # Update the sample history for analysis purpose
        if extra_info["is_new"]:
            self.history[self.n_sample_distinct] = self.strata._n_sampled_distinct

    def _calc_BB_prior(self, theta_0):
        """Generate a prior for the BB model

        Parameters
        ----------
        theta_0 : array-like, shape=(n_strata,)
            array of oracle probabilities (probability of a "1" label)
            for each stratum. This is just a guess.

        Returns
        -------
        alpha_0 : numpy.ndarray, shape=(n_strata,)
            "alpha" hyperparameters for an ensemble of Beta-distributed rvs

        beta_0 : numpy.ndarray, shape=(n_strata,)
            "beta" hyperparameters for an ensemble of Beta-distributed rvs
        """
        #: Easy vars
        prior_strength = self.prior_strength

        #weighted_strength = self.weights * strength
        n_strata = len(theta_0)
        weighted_strength = prior_strength / n_strata
        alpha_0 = theta_0 * weighted_strength
        beta_0 = (1 - theta_0) * weighted_strength
        return alpha_0, beta_0


    def reset(self):
        """Resets the sampler to its initial state

        Note
        ----
        This will destroy the label cache, instrumental distribution and
        history of estimates.
        """
        super(StratifiedSampler, self).reset()
        self.history = np.zeros_like(self.history)
        self.strata.reset()
        self._BB_model.reset()

