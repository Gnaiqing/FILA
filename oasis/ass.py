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

class ASSSampler(PassiveSampler):
    def __init__(self, alpha, predictions, scores, oracle, proba=False,
                 epsilon=1e-3, opt_class=None, prior_strength=None,
                 decaying_prior=True, strata=None, record_inst_hist=False,
                 max_iter=None, identifiers=None, debug=False, **kwargs):
        super(ASSSampler, self).__init__(alpha, predictions, oracle,
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