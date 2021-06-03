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
                 decaying_prior=False, strata=None, record_inst_hist=False,
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
            self.prior_strength = verify_positive(float(self.prior_strength))

        # Instantiate Beta-Bernoulli model using probabilities averaged over
        # opt_class
        # theta_0 = self.strata.intra_mean(self._probs_avg_opt_class)
        # gamma = self._calc_BB_prior(theta_0.ravel())
        # Instantiate Beta-Bernoulli model using uniform prior
        gamma = np.zeros((2,self.strata.n_strata_))
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

    def _calc_var_F_guess(self, tp_est, tp_var, fn_est, fn_var):
        """
        Estimate the variance of F-score estimator based on delta method
        F = tp / (p*alpha + (tp+fn)*(1-alpha))
        :param alpha: alpha for F score
        :param tp_est: estimation of TP
        :param fn_est: estimation of FN
        :param p: TP+FP (known constant)
        :return: var_F: estimated variance of F-score
        """
        p = self.P
        alpha = self.alpha
        partial_tp = (p*alpha + (1-alpha)*fn_est)/(p*alpha+(tp_est+fn_est)*(1-alpha))**2
        partial_fn = -(1-alpha)*tp_est/(p*alpha+(tp_est+fn_est)*(1-alpha))**2
        var_F = tp_var*partial_tp**2 + fn_var * partial_fn**2
        return var_F


    def select_stratum(self, sample_strategy, **kwargs):
        """
        Pick the next stratum to sample from according to sample_strategy
        :param sample_strategy: "ass"
        :return: stratum_idx: index of stratum selected
        """
        if sample_strategy == "ass":
            assert len(self.mix_strata_idx) == 0
            emp_strata_var = self._BB_model.calc_strata_var(include_prior=True)
            emp_strata_std = np.sqrt(emp_strata_var)
            tp_est = (self._BB_model.theta_*self.strata.sizes_)[self.pos_strata_idx].sum()
            fn_est = (self._BB_model.theta_*self.strata.sizes_)[self.neg_strata_idx].sum()
            p = self.P
            alpha = self.alpha
            partial_tp = (p*alpha + (1-alpha)*fn_est)/(p*alpha+(tp_est+fn_est)*(1-alpha))**2
            partial_fn = -(1-alpha)*tp_est/(p*alpha+(tp_est+fn_est)*(1-alpha))**2
            opt_strata_weight = np.zeros(self.strata.n_strata_)
            opt_strata_weight[self.pos_strata_idx] = np.abs(partial_tp * emp_strata_std * self.strata.sizes_)[self.pos_strata_idx]
            opt_strata_weight[self.neg_strata_idx] = np.abs(partial_fn * emp_strata_std * self.strata.sizes_)[self.neg_strata_idx]
            # select the current strata based on greedy search
            ucb = opt_strata_weight / self.strata._n_sampled
            stratum_idx = np.argmax(ucb)
            return stratum_idx

    def _sample_item(self, **kwargs):
        """Sample an item from the pool
        """
        if 'oracle' in kwargs and kwargs['oracle']:
            return self._sample_item_oracle(**kwargs)

        if 'fixed_stratum' in kwargs:
            stratum_idx = kwargs['fixed_stratum']
        else:
            # for each stratified sampling method, we first guarantee at least two points per strata
            if np.min(self.strata._n_sampled) < 2:
                stratum_idx = np.argmin(self.strata._n_sampled)
            elif 'sample_strategy' in kwargs:
                stratum_idx = self.select_stratum(kwargs["sample_strategy"], **kwargs)
            else:
                stratum_idx = self.select_stratum("ass", **kwargs)

        loc = self.strata._sample_in_stratum(stratum_idx,
                                                 replace=self.replace)

        return loc, 1, {'stratum': stratum_idx}


    def _sample_item_oracle(self, **kwargs):
        """
        Sample an item based on oracle (var and mean of each strata known in advance)
        :param kwargs:
        :return:
        """
        if 'fixed_stratum' in kwargs:
            stratum_idx = kwargs['fixed_stratum']
        else:
            if np.min(self.strata._n_sampled) < 2:
                stratum_idx = np.argmin(self.strata._n_sampled)
                loc = self.strata._sample_in_stratum(stratum_idx,
                                                     replace=self.replace)
                return loc, 1, {'stratum': stratum_idx}

            if 'sample_strategy' in kwargs:
                sample_strategy = kwargs["sample_strategy"]
            else:
                sample_strategy = "ass"

            if sample_strategy == "prop":
                sample_weight = self.strata.sizes_
            elif sample_strategy == "neyman":
                sample_weight = self.strata.sizes_ * np.sqrt(self.strata_var)
            else:
                sample_weight = self.strata.sizes_ * np.sqrt(self.strata_var) * self.partial_weight

            sample_weight = sample_weight / np.sum(sample_weight)
            loc, stratum_idx = self.strata.sample(pmf = sample_weight,
                                                  replace=self.replace)

            return loc, 1, {'stratum': stratum_idx}




    def _update_estimate_and_sampler(self, ell, ell_hat, weight, extra_info,
                                     **kwargs):
        # updating the BB model
        self._BB_model.update(ell, extra_info['stratum'])
        # update current estimation
        self._TP = (self._BB_model.theta_*self.strata.sizes_)[self.pos_strata_idx].sum()
        self._FN = (self._BB_model.theta_*self.strata.sizes_)[self.neg_strata_idx].sum()
        self._FP = self.P - self._TP
        self._estimate[self.t_] = \
            self._F_measure(self.alpha, self._TP, self._FP, self._FN)

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
        self.strata.reset()
        self._BB_model.reset()

