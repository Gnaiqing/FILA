import numpy as np
import random
import oasis
import matplotlib.pyplot as plt
import json
import argparse
import os
import pandas as pd


# def oracle(idx):
#     return data.labels[idx]


def plt_estimates(result_list, true_value, title, dataset_name):
    plt.figure()
    for result in result_list:
        plt.plot(result["estimate"],label=result["name"])
        if "est_std" in result:
            x = np.arange(len(result["estimate"]))
            y1 = result["estimate"] - 1.96* result["est_std"]
            y2 = result["estimate"] + 1.96 * result["est_std"]
            plt.fill_between(x,y1,y2 ,color='b',alpha=.1)

    plt.axhline(y=true_value, ls="--", color='k')
    plt.xlabel("Label budget")
    plt.ylabel("Estimate of F-measure")
    plt.ylim(0,1.05)
    plt.legend()
    plt.title(title)
    if not os.path.exists("fig/%s" % dataset_name):
        os.makedirs("fig/%s" % dataset_name, exist_ok=True)
    path = "fig/%s/%s.png" % (dataset_name, title)
    plt.savefig(path)


def plt_expt(result_list, title, dataset_name):
    fontsize = 14
    plt.figure(figsize=[4.8,6.4])
    plt.subplot(211)
    for result in result_list:
        plt.plot(result["abs_err"].reshape(-1),label=result["name"])
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
    plt.ylabel("abs_err",fontsize=fontsize)
    plt.title(title,fontsize=fontsize)
    plt.legend()

    plt.subplot(212)
    for result in result_list:
        std = np.sqrt(result["variance"]).reshape(-1)
        plt.plot(std,label=result["name"])
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
    plt.xlabel("label budget",fontsize=fontsize)
    plt.ylabel("std",fontsize=fontsize)
    if not os.path.exists("fig/%s" % dataset_name):
        os.makedirs("fig/%s" % dataset_name, exist_ok=True)
    path = "fig/%s/%s_stats.png" % (dataset_name, title)
    plt.savefig(path, bbox_inches="tight")


def plt_conf(results, title, dataset_name):
    fontsize = 14
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('label_budget', fontsize=fontsize)
    ax1.tick_params(axis='x', labelsize=fontsize)
    ax1.set_ylabel('interval_length',fontsize=fontsize)
    ax1.tick_params(axis='y',labelsize=fontsize)
    ax2 = ax1.twinx()
    ax2.set_ylabel('interval_accuracy',fontsize=fontsize)
    ax2.tick_params(axis='y',labelsize=fontsize)
    for result in results:
        ax1.plot(result["interval_length"], label=result["name"])
        ax2.plot(result["interval_accuracy"], "--", label=result["name"])
    plt.legend()
    path = "fig/%s/%s_ival.png" % (dataset_name,title)
    plt.savefig(path, bbox_inches="tight")


def single_run(dataset_name, data_path, config_path="exp_config.json", random_seed = 0,
               alpha=0.5, n_labels=5000, max_iter=100000, name_list=None, exp_tag=""):
    print("Working on dataset %s" % dataset_name)
    np.random.seed(random_seed)
    random.seed(random_seed)
    data = oasis.Data()
    suffix = data_path.split(".")[-1]
    if suffix == "h5":
        data.read_h5(data_path)
    elif suffix == "csv":
        data.read_csv(data_path)
    else:
        raise Exception("File format not supported: %s" % suffix)

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
            if obj["name"] == "FILA-thompson":
                # print confidence interval for FILA method
                result["est_std"] = smplr.estimate_std[smplr.queried_oracle_]
            result_list.append(result)

    if len(result_list) > 0:
        plt_estimates(result_list, data.F_measure, dataset_name+"_"+exp_tag, dataset_name)


def multiple_run(dataset_name, data_path, config_path="exp_config.json",
                 random_seed = 0, n_expts = 100,
                 alpha=0.5, n_labels=5000, max_iter=1000000, name_list=None,
                 restore=False, exp_tag = "", result_file = "results.csv"):
    print("Working on dataset %s" % dataset_name)
    np.random.seed(random_seed)
    random.seed(random_seed)
    # store result in a dataframe
    result_df = pd.DataFrame(
        {
            "dataset" : [],
            "sampler": [],
            "alpha": [],
            "n_labels": [],
            "abs_err":[],
            "std_dev":[],
            "time":[]
        }
    )
    data = oasis.Data()
    suffix = data_path.split(".")[-1]
    if suffix == "h5":
        data.read_h5(data_path)
    elif suffix == "csv":
        data.read_csv(data_path)
    else:
        raise Exception("File format not supported: %s" % suffix)
    data.calc_true_performance(printout=True, alpha=alpha)
    positive_idx = np.where(data.preds == 1)
    positive_scores = data.scores[positive_idx]
    threshold = np.min(positive_scores)
    result_list = []
    # result_list_with_interval = []
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

            output_file = "exp/d=%s_s=%s_a=%.2f_%d.h5" % \
                          (dataset_name, obj["name"], alpha, random_seed)
            if not restore or not os.path.exists(output_file):
                oasis.repeat_expt(smplr, n_expts, n_labels, output_file, **sample_config)
            result = oasis.process_expt(output_file, F_gt = data.F_measure)
            result["name"] = obj["name"]
            print(result["name"], "mean_time:", result["mean_CPU_time"], "iterations:", result["mean_n_iterations"])
            result_list.append(result)
            # show results qualitatively
            print("Evaluating ", result["name"])
            print("Abs. Err:", result["abs_err"].reshape(-1)[-1])
            print("Std. Dev:", result["variance"].reshape(-1)[-1])
            if hasattr(smplr, "estimate_std"):
                print("conf interval accuracy:", result["interval_accuracy"][-1])
                print("conf interval length  :", result["interval_length"][-1])

            result_df = result_df.append(
                {
                    "dataset" : dataset_name,
                    "sampler": result["name"],
                    "alpha": alpha,
                    "n_labels": n_labels,
                    "abs_err":result["abs_err"].reshape(-1)[-1],
                    "std_dev":result["variance"].reshape(-1)[-1],
                    "time":result["mean_CPU_time"]
                }, ignore_index=True
            )
            #     result_list_with_interval.append(result)

    if len(result_list) > 0:
        title = "{}_{}".format(dataset_name, exp_tag)
        plt_expt(result_list,title, dataset_name)

    # store data to result file
    with open(result_file,"a") as f:
        result_df.to_csv(f, header=f.tell() == 0, line_terminator='\n')
    # if len(result_list_with_interval) > 0:
    #     plt_conf(result_list_with_interval, dataset_name, dataset_name)


def synthetic_experiment(hp, n_labels):
    # conduct experiment on synthetic data
    for size in ["1e+05"]:
        for imbalance_ratio in ["10","30","100"]:
            for sigma in ["1","5","25"]:
                for classifier in ["mlp","svm"]:
                    dataset_name = "syn-sz=%s-imb=%s-sigma=%s-%s" % \
                        (size, imbalance_ratio, sigma, classifier)
                    dataset_path = "datasets/synthetic/%s.csv" % dataset_name
                    if os.path.exists(dataset_path):
                        multiple_run(dataset_name,dataset_path,
                                     alpha=hp.alpha,
                                     n_labels=n_labels,
                                     max_iter=hp.max_iter,
                                     n_expts=hp.n_expts,
                                     name_list=hp.smplr_list,
                                     restore=hp.restore,
                                     exp_tag = hp.exp_tag,
                                     random_seed=hp.random_seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg_path", type=str, default="dataset_config.json")
    parser.add_argument("--dataset_list", type=str, nargs="+", default=None)
    parser.add_argument("--smplr_cfg_path", type=str, default="exp_config.json")
    parser.add_argument("--smplr_list",type=str, nargs="+",default=["OASIS","Proportional","Neyman-thompson","FILA-thompson"])
    parser.add_argument("--exp_type", type=str, choices=["single","stats"], default="single")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--exp_tag", type=str, default="")
    parser.add_argument("--n_expts", type=int, default=100)
    parser.add_argument("--n_labels",type=int, default=None)
    parser.add_argument("--max_iter",type=int, default=1000000)
    parser.add_argument("--restore", dest="restore", action="store_true")
    hp = parser.parse_args()
    hp.exp_tag = chr(945) + "=" + str(hp.alpha)
    data_config = json.load(open(hp.data_cfg_path))
    for dataset in data_config:
        dataset_name = dataset["name"]
        dataset_path = dataset["path"]
        if hp.n_labels is not None:
            n_labels = hp.n_labels
        elif "n_labels" in dataset:
            n_labels = dataset["n_labels"]
        else:
            n_labels = 300

        if hp.dataset_list is not None and dataset_name not in hp.dataset_list:
            continue
        if hp.exp_type == "single":
            single_run(dataset_name, dataset_path,
                       alpha=hp.alpha,
                       n_labels = n_labels,
                       max_iter=hp.max_iter,
                       name_list = hp.smplr_list,
                       exp_tag = hp.exp_tag,
                       random_seed=hp.random_seed)
        else:
            multiple_run(dataset_name,dataset_path,
                         alpha=hp.alpha,
                         n_labels=n_labels,
                         max_iter=hp.max_iter,
                         n_expts=hp.n_expts,
                         name_list=hp.smplr_list,
                         restore=hp.restore,
                         exp_tag = hp.exp_tag,
                         random_seed=hp.random_seed)
    if "synthetic" in hp.dataset_list:
        synthetic_experiment(hp, n_labels)


