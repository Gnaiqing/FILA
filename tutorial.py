import oasis
import numpy as np
import random
import oasis
import matplotlib.pyplot as plt


def oracle(idx):
    return data.labels[idx]

def plt_estimates(smplr, true_value, title):
    plt.figure()
    plt.plot(smplr.estimate_[smplr.queried_oracle_])
    plt.axhline(y=true_value, color='r')
    plt.xlabel("Label budget")
    plt.ylabel("Estimate of F1-score")
    plt.ylim(0,1)
    plt.title(title)
    path = "fig/" + title + ".png"
    plt.savefig(path)


if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    data = oasis.Data()
    data.read_h5('docs/tutorial/Amazon-GoogleProducts-test.h5')
    data.calc_true_performance() #: calculate true precision, recall, F1-score
    positive_idx = np.where(data.preds == 1)
    positive_scores = data.scores[positive_idx]
    threshold = np.min(positive_scores)
    alpha = 0.5      #: corresponds to F1-score
    n_labels = 5000  #: stop sampling after querying this number of labels
    max_iter = 100000   #: maximum no. of iterations that can be stored
    ss_smplr = oasis.StratifiedSampler(alpha, data.preds, data.scores, oracle,
                                       max_iter=max_iter,
                                       stratification_threshold=threshold,
                                       labels=data.labels)
    oasis_smplr = oasis.OASISSampler(alpha, data.preds, data.scores, oracle, max_iter=max_iter,stratification_threshold=threshold)
    # druck_smplr = oasis.DruckSampler(alpha, data.preds, data.scores, oracle, max_iter=max_iter,stratification_threshold=threshold)

    for sample_strategy in ["prop","neyman", "ass"]:
        ss_smplr.sample_distinct(n_labels, oracle=True, sample_strategy=sample_strategy)
        plt_estimates(ss_smplr, data.F1_measure,sample_strategy+" oracle")
        ss_smplr.reset()

    oasis_smplr.sample_distinct(n_labels)
    plt_estimates(oasis_smplr, data.F1_measure, "oasis")
