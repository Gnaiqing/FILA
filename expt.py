import numpy as np
import random
import oasis
import matplotlib.pyplot as plt
import json
import argparse
import os


# def oracle(idx):
#     return data.labels[idx]


def plt_estimates(result_list, true_value, title):
    plt.figure()
    for result in result_list:
        plt.plot(result["estimate"],label=result["name"])

    plt.axhline(y=true_value, ls="--", color='k')
    plt.xlabel("Label budget")
    plt.ylabel("Estimate of F-score")
    plt.ylim(0,1)
    plt.legend()
    plt.title(title)
    path = "fig/" + title + ".png"
    plt.savefig(path)


def plt_expt(result_list, title):
    plt.figure()
    plt.subplot(211)
    for result in result_list:
        plt.plot(result["abs_err"].reshape(-1),label=result["name"])
    plt.ylabel("abs_err")
    plt.title(title)
    plt.legend()

    plt.subplot(212)
    for result in result_list:
        std = np.sqrt(result["variance"]).reshape(-1)
        plt.plot(std,label=result["name"])
    plt.xlabel("n_labels")
    plt.ylabel("std")

    path = "fig/" + title + "_stats.png"
    plt.savefig(path)


def single_run(dataset_name, data_path, config_path="exp_config.json", random_seed = 0,
               alpha=0.5, n_labels=5000, max_iter=100000, name_list=None, exp_tag=""):
    print("Working on dataset %s" % dataset_name)
    np.random.seed(random_seed)
    random.seed(random_seed)
    data = oasis.Data()
    data.read_h5(data_path)
    data.calc_true_performance(printout=True, alpha=alpha)
    positive_idx = np.where(data.preds == 1)
    positive_scores = data.scores[positive_idx]
    threshold = np.min(positive_scores)
    result_list = []
    config = json.load(open(config_path))
    oracle = lambda idx: data.labels[idx]
    for obj in config:
        if name_list is None or obj["name"] in name_list:
            print("Evaluating sampler %s" % obj["name"])
            if "smplr_config" in obj:
                smplr_config = obj["smplr_config"]
            else:
                smplr_config = {}

            if obj["smplr"] == "OASIS":
                smplr = oasis.OASISSampler(alpha, data.preds, data.scores, oracle,
                                           max_iter=max_iter,
                                           stratification_threshold=threshold,
                                           labels=data.labels,
                                           **smplr_config)
            elif obj["smplr"] == "SS":
                smplr = oasis.StratifiedSampler(alpha, data.preds, data.scores, oracle,
                                                max_iter=max_iter,
                                                stratification_threshold=threshold,
                                                labels=data.labels,
                                                **smplr_config)
            else:
                print("sampler %s not supported" % obj["smplr"])
                continue

            if "sample_config" in obj:
                sample_config = obj["sample_config"]
            else:
                sample_config = {}

            smplr.sample_distinct(n_labels, **sample_config)
            result = {
                "name" : obj["name"],
                "estimate" : smplr.estimate_[smplr.queried_oracle_]
            }
            result_list.append(result)

    if len(result_list) > 0:
        plt_estimates(result_list, data.F_measure, dataset_name+exp_tag)


def multiple_run(dataset_name, data_path, config_path="exp_config.json",
                 random_seed = 0, n_expts = 100,
                 alpha=0.5, n_labels=5000, max_iter=1000000, name_list=None,
                 restore=False, exp_tag = ""):
    print("Working on dataset %s" % dataset_name)
    np.random.seed(random_seed)
    random.seed(random_seed)
    data = oasis.Data()
    data.read_h5(data_path)
    data.calc_true_performance(printout=True, alpha=alpha)
    positive_idx = np.where(data.preds == 1)
    positive_scores = data.scores[positive_idx]
    threshold = np.min(positive_scores)
    result_list = []
    config = json.load(open(config_path))
    oracle = lambda idx: data.labels[idx]
    for obj in config:
        if name_list is None or obj["name"] in name_list:
            print("Evaluating sampler %s" % obj["name"])
            if "smplr_config" in obj:
                smplr_config = obj["smplr_config"]
            else:
                smplr_config = {}
            if obj["smplr"] == "OASIS":
                smplr = oasis.OASISSampler(alpha, data.preds, data.scores, oracle,
                                           max_iter=max_iter,
                                           stratification_threshold=threshold,
                                           labels=data.labels,
                                           **smplr_config)
            elif obj["smplr"] == "SS":
                smplr = oasis.StratifiedSampler(alpha, data.preds, data.scores, oracle,
                                                max_iter=max_iter,
                                                stratification_threshold=threshold,
                                                labels=data.labels,
                                                **smplr_config)
            else:
                print("sampler %s not supported" % obj["smplr"])
                continue

            if "sample_config" in obj:
                sample_config = obj["sample_config"]
            else:
                sample_config = {}

            output_file = "exp/d=%s_s=%s_%d.h5" % \
                          (dataset_name, obj["name"], random_seed)
            if not restore or not os.path.exists(output_file):
                oasis.repeat_expt(smplr, n_expts, n_labels, output_file, **sample_config)
            result = oasis.process_expt(output_file, F_gt = data.F_measure)
            result["name"] = obj["name"]
            result_list.append(result)

    if len(result_list) > 0:
        plt_expt(result_list,dataset_name+exp_tag)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg_path", type=str, default="dataset_config.json")
    parser.add_argument("--dataset_list", type=str, nargs="+", default=None)
    parser.add_argument("--smplr_cfg_path", type=str, default="exp_config.json")
    parser.add_argument("--exp_type", type=str, choices=["single","stats"], default="single")
    parser.add_argument("--exp_tag", type=str, default="")
    parser.add_argument("--n_expts", type=int, default=100)
    parser.add_argument("--restore", dest="restore", action="store_true")
    hp = parser.parse_args()
    data_config = json.load(open(hp.data_cfg_path))
    for dataset  in data_config:
        dataset_name = dataset["name"]
        dataset_path = dataset["path"]
        if hp.dataset_list is not None and dataset_name not in hp.dataset_list:
            continue
        if hp.exp_type == "single":
            single_run(dataset_name, dataset_path,
                       name_list = ["OASIS", "SS-prop", "SS-neyman", "SS-opt"],
                       exp_tag = hp.exp_tag)
        else:
            multiple_run(dataset_name,dataset_path,n_expts=hp.n_expts,
                         name_list=["SS-neyman", "SS-neyman-oracle", "SS-opt", "SS-opt-oracle"],
                         restore=hp.restore,
                         exp_tag = hp.exp_tag)

