import oasis
import numpy as np
import random
import oasis
import matplotlib.pyplot as plt


def oracle(idx):
    return data.labels[idx]

def plt_estimates(smplr, true_value):
    plt.plot(smplr.estimate_[smplr.queried_oracle_])
    plt.axhline(y=true_value, color='r')
    plt.xlabel("Label budget")
    plt.ylabel("Estimate of F1-score")
    plt.ylim(0,1)
    plt.show()

if __name__ == "__main__":
    np.random.seed(319158)
    random.seed(319158)
    data = oasis.Data()
    data.read_h5('docs/tutorial/Amazon-GoogleProducts-test.h5')
    data.calc_true_performance() #: calculate true precision, recall, F1-score
    alpha = 0.5      #: corresponds to F1-score
    n_labels = 500  #: stop sampling after querying this number of labels
    max_iter = 1000   #: maximum no. of iterations that can be stored
    smplr = oasis.OASISSampler(alpha, data.preds, data.scores, oracle, max_iter=max_iter)
    smplr.sample_distinct(n_labels)
    plt_estimates(smplr, data.F1_measure)
