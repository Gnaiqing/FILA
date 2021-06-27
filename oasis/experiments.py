import numpy as np
import tables
import time
import logging
import warnings
import pandas as pd

def repeat_expt(smplr, n_expts, n_labels, output_file = None, **kwargs):
    """
    Parameters
    ----------
    smplr : sub-class of PassiveSampler
        sampler must have a sample_distinct method, reset method and ...

    n_expts : int
        number of expts to run

    n_labels : int
        number of labels to query from the oracle in each expt
    """

    FILTERS = tables.Filters(complib='zlib', complevel=5)

    max_iter = smplr._max_iter
    n_class = smplr._n_class
    if max_iter < n_labels:
        raise ValueError("Cannot query {} labels. Sampler ".format(n_labels) +
                         "instance supports only {} iterations".format(max_iter))

    if output_file is None:
        # Use current date/time as filename
        output_file = 'expt_' + time.strftime("%d-%m-%Y_%H:%M:%S") + '.h5'
    logging.info("Writing output to {}".format(output_file))

    f = tables.open_file(output_file, mode='w', filters=FILTERS)
    float_atom = tables.Float64Atom()
    bool_atom = tables.BoolAtom()
    int_atom = tables.Int64Atom()

    array_F = f.create_carray(f.root, 'F_measure', float_atom, (n_expts, n_labels, n_class))
    array_std = f.create_carray(f.root,'F_std', float_atom, (n_expts, n_labels, n_class))
    array_s = f.create_carray(f.root, 'n_iterations', int_atom, (n_expts, 1))
    array_t = f.create_carray(f.root, 'CPU_time', float_atom, (n_expts, 1))

    logging.info("Starting {} experiments".format(n_expts))
    for i in range(n_expts):
        if i%np.ceil(n_expts/10).astype(int) == 0:
            logging.info("Completed {} of {} experiments".format(i, n_expts))
        ti = time.process_time()
        smplr.reset()
        smplr.sample_distinct(n_labels, **kwargs)
        tf = time.process_time()
        if hasattr(smplr, 'queried_oracle_'):
            array_F[i,:,:] = smplr.estimate_[smplr.queried_oracle_].reshape(-1,n_class)
            if hasattr(smplr, 'estimate_std'):
                array_std[i,:,:] = smplr.estimate_std[smplr.queried_oracle_].reshape(-1,n_class)
        else:
            array_F[i,:,:] = smplr.estimate_.reshape(-1,n_class)
            if hasattr(smplr, 'estimate_std'):
                array_std[i,:,:] = smplr.estimate_std.reshape(-1,n_class)
        array_s[i] = smplr.t_
        array_t[i] = tf - ti
    f.close()

    logging.info("Completed all experiments")


def process_expt(h5_path, inmemory = True, ignorenan = False, F_gt = None):
    """
    Assumes h5 file has table called `F_measure`

    Parameters
    ----------
    h5_path : string
        path to HDF file containing the experimental data. The file is expected
        to have been generated from the `repeat_expt` function.

    inmemory : bool
        whether to process the experiments in memory

    ignorenan : bool
        whether to ignore NaNs when computing the mean and variance
    """

    logging.info("Reading file at {}".format(h5_path))
    h5_file = tables.open_file(h5_path, mode = 'r')

    F = h5_file.root.F_measure
    # F_std = h5_file.root.F_std

    n_expt, n_labels, n_class = F.shape
    mean_n_iterations = np.sum(h5_file.root.n_iterations)/n_expt

    if hasattr(h5_file.root, 'CPU_time'):
        CPU_time = h5_file.root.CPU_time
        mean_CPU_time = np.mean(CPU_time)
        var_CPU_time = np.var(CPU_time)
    else:
        mean_CPU_time = None
        var_CPU_time = None
        mean_CPU_time_per_iteration = None

    F_mean = np.empty([n_labels, n_class], dtype='float')
    F_var = np.empty([n_labels, n_class], dtype='float')
    F_stderr = np.empty([n_labels, n_class], dtype='float')
    F_abserr = np.empty([n_labels, n_class], dtype='float')
    n_sample = np.empty(n_labels, dtype='int')
    # store the accuracy and length of confidence interval
    I_mean_length = np.empty([n_labels, n_class], dtype='float')
    I_accuracy = np.empty([n_labels, n_class], dtype='float')

    if inmemory:
        F_mem = F[:,:,:]
        # F_mem_std = F_std[:,:,:]

    logging.info("Beginning processing".format())
    for t in range(n_labels):

        if t%np.ceil(n_labels/10).astype(int) == 0:
            logging.info("Processed {} of {} experiments".format(t, n_labels))
        if inmemory:
            temp = F_mem[:,t,:]
            # temp_std = F_mem_std[:,t,:]
        else:
            temp = F[:,t,:]
            # temp_std = F_std[:,t,:]
        if ignorenan:
            n_sample[t] = np.sum(~np.isnan(temp))
            # Expect to see RuntimeWarnings if array contains all NaNs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                F_mean[t] = np.nanmean(temp, axis=0)
                F_var[t] = np.nanvar(temp, axis=0)
                F_stderr[t] = np.sqrt(F_var[t]/n_sample[t])
                # I_mean_length[t] = min(1, np.nanmean(temp_std, axis=0) * 1.96 * 2)
                if F_gt:
                    F_abserr[t] = np.nanmean(np.abs(temp-F_gt), axis=0)
                    # compute accuracy of confidence interval
                    # lb = temp - 1.96*temp_std
                    # ub = temp + 1.96*temp_std
                    # I_accuracy[t] = np.nanmean((lb<=F_gt)& (F_gt<=ub), axis=0)


        else:
            n_sample[t] = len(temp)
            F_mean[t] = np.mean(temp, axis=0)
            F_var[t] = np.var(temp, axis=0)
            F_stderr[t] = np.sqrt(F_var[t]/n_sample[t])
            # I_mean_length[t] = min(1, np.mean(temp_std, axis=0) * 1.96 * 2)
            if F_gt:
                F_abserr[t] = np.mean(np.abs(temp-F_gt), axis=0)
                # compute accuracy of confidence interval
                # lb = temp - 1.96*temp_std
                # lb = np.nan_to_num(lb, nan=0)
                # ub = temp + 1.96*temp_std
                # ub = np.nan_to_num(ub, nan=1)
                # I_accuracy[t] = np.mean((lb<=F_gt)& (F_gt<=ub), axis=0)

    logging.info("Processing complete".format())

    h5_file.close()

    return {'mean': F_mean,
            'variance': F_var,
            'std_error': F_stderr,
            'n_samples': n_sample,
            'n_expts': n_expt,
            'n_labels': n_labels,
            'mean_CPU_time': mean_CPU_time,
            'var_CPU_time': var_CPU_time,
            'mean_n_iterations': mean_n_iterations,
            'h5_path': h5_path,
            'abs_err': F_abserr
            # 'interval_length': I_mean_length,
            # 'interval_accuracy': I_accuracy
            }


class DataError(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg

class Data:
    """

    """
    def __init__(self):
        self.features = None
        self.scores = None
        self.labels = None
        self.probs = None
        self.preds = None
        self.threshold = None
        self.TP = None
        self.FP = None
        self.TN = None
        self.FN = None
        self.F_measure = None
        self.precision = None
        self.recall = None
        self.ratio_factor = None

    def read_csv(self, csv_path):
        print("Loading %s" % csv_path)
        df = pd.read_csv(csv_path)
        columns = df.columns
        if "labels" in columns:
            self.labels = df["labels"].to_numpy()
        if "probs" in columns:
            self.probs = df["probs"].to_numpy()
        if "scores" in columns:
            self.scores = df["scores"].to_numpy()
        if "preds" in columns:
            self.preds = df["preds"].to_numpy()

        from scipy.special import expit,logit
        if self.probs is None and self.scores is None:
            raise RuntimeError('probs and scores both missing')
        if self.probs is None:
            warnings.warn('converting scores into probabilities', UserWarning)
            self.probs = expit(self.scores)
        if self.scores is None:
            warnings.warn('converting probabilities into scores', UserWarning)
            self.scores = logit(self.probs)
        if self.preds is None:
            warnings.warn('making predictions from scores using default threshold of zero', UserWarning)
            self.preds = (self.scores >= 0) * 1

    def read_h5(self, h5_path, load_features=False):
        print("Loading %s" % h5_path)
        h5_file = tables.open_file(h5_path, mode = 'r')

        if load_features and hasattr(h5_file.root, "features"):
            self.features = h5_file.root.features[:,:]
            self.num_fts = h5_file.root.features.shape[1]
        if hasattr(h5_file.root, "labels"):
            self.labels = h5_file.root.labels[:]
            self.num_items = len(self.labels)
        if hasattr(h5_file.root, "probs"):
            self.probs = h5_file.root.probs[:]
            self.num_items = len(self.probs)
        if hasattr(h5_file.root, "scores"):
            self.scores = h5_file.root.scores[:]
            self.num_items = len(self.scores)
        if hasattr(h5_file.root, "preds"):
            self.preds = h5_file.root.preds[:]
            self.num_items = len(self.preds)

        h5_file.close()
        from scipy.special import expit,logit
        if h5_path.lower().find('svm') != -1:
            # Remove calibrated probabilities from SVM which were expensive to compute
            self.probs = None
        if self.probs is None and self.scores is None:
            raise RuntimeError('probs and scores both missing')
        if self.probs is None:
            warnings.warn('converting scores into probabilities', UserWarning)
            self.probs = expit(self.scores)
        if self.scores is None:
            warnings.warn('converting probabilities into scores', UserWarning)
            self.scores = logit(self.probs)
        if self.preds is None:
            warnings.warn('making predictions from scores using default threshold of zero', UserWarning)
            self.preds = (self.scores >= 0) * 1

    def scores_to_preds(self, threshold, use_probs = True):
        """
        use_probs : boolean, default True
            if True, use probabilities for predictions, else use scores.
        """
        self.threshold = threshold

        if use_probs:
            if self.probs is None:
                raise DataError("Probabilities are not available to make "
                                "predictions.")
            else:
                word = "probabilities"
                scores = self.probs
        else:
            if self.scores is None:
                raise DataError("Scores are not available to make predictions.")
            else:
                word = "scores"
                scores = self.scores

        if threshold > np.max(scores) or threshold < np.min(scores):
            warnings.warn("Threshold {} is outside the range of the "
                          "{}.".format(self.threshold, word))

        if self.preds is not None:
            warnings.warn("Overwriting predictions")
        self.preds = (scores >= threshold)*1

    def calc_confusion_matrix(self, printout = False):
        """
        Calculates number of TP, FP, TN, FN
        """
        if self.labels is None:
            raise DataError("Cannot calculate confusion matrix before data "
                            "has been read.")

        if self.preds is None:
            raise DataError("Predictions not available. Please run "
                            "`scores_to_preds` before calculating confusion "
                            "matrix")

        self.TP = np.sum(np.logical_and(self.preds == 1, self.labels == 1))
        self.TN = np.sum(np.logical_and(self.preds == 0, self.labels == 0))
        self.FP = np.sum(np.logical_and(self.preds == 1, self.labels == 0))
        self.FN = np.sum(np.logical_and(self.preds == 0, self.labels == 1))

        if printout:
            print("Contingency matrix is:")
            print("----------------------")
            print("TP: {} \t FN: {}".format(self.TP,self.FN))
            print("FP: {} \t TN: {}".format(self.FP,self.TN))
            print("\n")


    def calc_true_performance(self, alpha = 0.5, printout = False):
        """
        Evaluate precision, recall and balanced F-measure
        """
        try:
            self.calc_confusion_matrix(printout = False)
        except DataError as e:
            print(e.msg)
            raise

        if self.TP + self.FP == 0:
            self.precision = np.nan
        else:
            self.precision = self.TP / (self.TP + self.FP)

        if self.TP + self.FN == 0:
            self.recall = np.nan
        else:
            self.recall = self.TP / (self.TP + self.FN)

        if self.precision + self.recall == 0:
            self.F_measure = np.nan
        else:
            self.F_measure = (self.precision * self.recall /
                              (alpha * self.recall + (1-alpha) * self.precision))

        if printout:
            print("True performance is:")
            print("--------------------")
            size = self.TP+self.FP+self.FN + self.TN
            pos  = self.TP+self.FP
            print("Size: {} Pos: {} Imb: {:.2f}".format(size, pos, (size-pos)/pos))
            print("TP: {}\t FP: {}\t FN: {}\t TN: {}".format(self.TP, self.FP, self.FN, self.TN) )
            print("Precision: {} \t Recall: {} \t F measure: {}".format(self.precision, self.recall, self.F_measure))
            self.ratio_factor = (alpha*(self.TP+self.FP)+(1-alpha)*(self.FN))/((1-alpha)*(self.TP))
            print("Ratio: {}".format(self.ratio_factor))