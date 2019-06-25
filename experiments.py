import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ttest_rel as ttest
from sim_normvec import VectorSimulator, euclidist
from sim_parser import ParserSimulator, ParsableStr, evalb
from sim_ranker import RankerSimulator, kendaltauscore
from simulation import BetaDist
import utils

def summary(sim):
    plt.hist(sim.err_rates)
    plt.title("True user errors")
    plt.show()
    plt.hist(list(sim.difficulty_dict.values()))
    plt.title("True item difficulties")
    plt.show()

def stan_summary(stan_data):
    print("NUSERS", stan_data["NUSERS"])
    print("NITEMS", stan_data["NITEMS"])

def params_baseline(sim, stan_data):
    pred_uerr_baseline = uerr_baseline(stan_data)[stan_data["n_gold_users"]:]
    scatter_corr(pred_uerr_baseline, sim.err_rates[stan_data["n_gold_users"]:], title="User Diagnostic")
    pred_diff_baseline = diff_baseline(stan_data)
    scatter_corr(pred_diff_baseline, list(sim.difficulty_dict.values()), title="Item Diagnostic")

def params_model(sim, opt, stan_data, skip_gold=False):
    start = stan_data["n_gold_users"] if skip_gold else 0
    scatter_corr(opt["uerr"][start:], sim.err_rates[start:], title="User baseline")
    scatter_corr(opt.get("disc"), list(sim.difficulty_dict.values()), title="Item baseline")

def uerr_baseline(stan_data):
    sddf = pd.DataFrame(stan_data)
    s1 = sddf.groupby("u1s").sum()["distances"]
    s2 = sddf.groupby("u2s").sum()["distances"]
    n1 = sddf.groupby("u1s").count()["distances"]
    n2 = sddf.groupby("u2s").count()["distances"]
    avg_distances = s1.add(s2, fill_value=0) / n1.add(n2, fill_value=0)
    return avg_distances

def diff_baseline(stan_data):
    sddf = pd.DataFrame(stan_data)
    avg_distances = sddf.groupby("items").mean()["distances"]
    return avg_distances

def scatter_corr(pred_vals, true_vals, jitter=False, title=None, log=False):
    if len(pred_vals) == 0:
        return
    if len(pred_vals) < len(true_vals):
        true_vals = np.array(true_vals)[pred_vals.index - 1]
    noise = lambda: np.random.uniform(-.5, .5, len(pred_vals)) if jitter else 0
    if len(pred_vals) < 1000:
        plt.scatter(np.array(pred_vals) + noise(), np.array(true_vals) + noise())
    else:
        plt.scatter(np.array(pred_vals) + noise(), np.array(true_vals) + noise(), s=1)
    if title is not None:
        plt.title(title)
    if log:
        plt.xscale("log")
        plt.yscale("log")
    plt.show()
    print(np.corrcoef(pred_vals, true_vals))

def pred_item(df, pred_uerr, label_colname, item_colname, uid_name="uid"):
    df["pred_uerr"] = pred_uerr[df[uid_name].values]
    def pickbybestuser(data):
        best_i = np.argmin(data["pred_uerr"].values)
        return data[label_colname].values[best_i]
    return df.groupby(item_colname).apply(pickbybestuser)

def pred_item_agg_exclude(df, pred_uerr, label_name, itemId_name, uid_name="uid", uerr_thresh_pctile=0.9):
    df["pred_uerr"] = pred_uerr[df[uid_name].values]
    err_thresh = np.quantile(pred_uerr, uerr_thresh_pctile)
    def mean_exclude(data):
        okdata = data[data["pred_uerr"] <= err_thresh]
        return okdata.mean()[label_name] if len(okdata) else data.mean()[label_name]
    return df.groupby(itemId_name).apply(mean_exclude)

def get_model_user_rankings(opt, debug=False):
    errs = opt["dist_from_truth"]
    result = np.argsort(errs, axis=1)
    tmp = errs[0][result[0]]
    assert tmp[0] <= tmp[1]
    if debug:
        errs[errs==666] = np.max(errs[errs<666]) * 1.1 # reset high-error value for better viz
        plt.imshow(errs.T, cmap='coolwarm', interpolation='nearest')
        plt.xlabel("items")
        plt.ylabel("users")
        plt.show()
    return result

def get_baseline_random(sim_df, label_colname, item_colname):
    def pickrandomlabel(data):
        return data.sample(1)[label_colname].values[0]
    return dict(sim_df.groupby(item_colname).apply(pickrandomlabel))

def get_baseline_global_best_user(stan_data, sim_df, label_colname, item_colname, uid_colname="uid"):
    uerrs = uerr_baseline(stan_data).values
    return dict(pred_item(sim_df, uerrs, label_colname, item_colname, uid_colname))

def get_baseline_item_centrallest(stan_data, sim_df, label_colname, item_colname, uid_colname="uid",
                                    agg_fn=None):
    sddf = pd.DataFrame(stan_data)
    preds = {}
    for i0 in sorted(sim_df[item_colname].unique()):
        i = i0 + 1
        iddf = sddf[sddf["items"] == i]
        if iddf.empty:
            idf = sim_df[sim_df[item_colname] == i0]
            pred = idf[label_colname].values[0]
        else:
            uerrs = uerr_baseline(iddf)
            i_sim_df = sim_df[sim_df[item_colname] == i-1]
            if agg_fn is None:
                best_user = uerrs.idxmin()
                pred = i_sim_df[i_sim_df[uid_colname] == best_user-1][label_colname].values[0]
            else:
                pred = agg_fn(i_sim_df, uerrs)
        preds[i-1] = pred
    return preds

def get_oracle_preds(stan_data, sim_df, label_colname, item_colname, uid_colname="uid",
                    eval_fn=None, gold_dict={}):
    if eval_fn is None:
        raise ValueError("Need a evaluation function to compute oracle")
    def agg_fn(i_sim_df, uerrs):
        item = i_sim_df[item_colname].values[0]
        gold = gold_dict.get(item)
        if gold is None:
            return i_sim_df[label_colname].values[0]
        evals = [eval_fn(gold, label) for label in i_sim_df[label_colname].values]
        pred = i_sim_df[label_colname].values[np.argmax(evals)]
        return pred
    return get_baseline_item_centrallest(stan_data, sim_df, label_colname, item_colname, uid_colname, agg_fn)

def get_preds(sim_df, per_item_user_rankings, label_colname, item_colname, user_colname="uid"):
    preds = {}
    for i, item in enumerate(sorted(sim_df[item_colname].unique())):
        idf = sim_df[sim_df[item_colname] == item]
        pred = None
        for best_user in per_item_user_rankings[i]:
            uidf = idf[idf[user_colname]==best_user]
            if len(uidf) > 0:
                pred = uidf[label_colname].values[0]
                break
        preds[item] = pred
    return preds

def eval_scores_vs(baseline_scores, model_scores, badness_threshold):
    diffs = np.array(model_scores) - np.array(baseline_scores)
    print(np.mean(baseline_scores), np.mean(model_scores))
    print("t-test", ttest(baseline_scores, model_scores))
    print("z-score", np.mean(diffs) / np.std(diffs))
    maxx = np.max(np.abs(diffs))
    print("baseline below thresh", (np.array(baseline_scores) < badness_threshold).mean())
    print("model below thresh", (np.array(model_scores) < badness_threshold).mean())
    # plt.hist(diffs, bins=np.linspace(-maxx, maxx, 10))
    # plt.show()

def eval_preds(preds, golds, eval_fn):
    scores = []
    for i, gold in golds.items():
        score = eval_fn(gold, preds[i]) if preds.get(i) is not None else 0
        scores.append(score)
    return scores

def eval_preds_vs(baseline_preds, model_preds, golds, eval_fn, print_diffs=False, badness_threshold=0):
    baseline_scores = eval_preds(baseline_preds, golds, eval_fn)
    model_scores = eval_preds(model_preds, golds, eval_fn)
    eval_scores_vs(baseline_scores, model_scores, badness_threshold)
    return baseline_scores, model_scores

class Experiment():
    def __init__(self, label_colname, item_colname, uid_name="uid"):
        self.label_colname = label_colname
        self.item_colname = item_colname
        self.uid_name = uid_name
        self.stan_data = None
        self.opt = None
        self.simulator = None
        self.badness_threshold = 0
    def train(self, use_diff=0, use_disc=1, dim_size=8, iter=500, **kwargs):
        if self.stan_data is None:
            raise ValueError("Must setup stan_data first")
        stan_model = utils.stanmodel("distance5", overwrite=False)
        self.stan_data["use_diff"] = use_diff
        self.stan_data["use_disc"] = use_disc
        self.stan_data["use_norm"] = 1
        self.stan_data["norm_ratio"] = 0.5
        self.stan_data["DIM_SIZE"] = dim_size
        self.stan_data["eps_limit"] = 1
        self.stan_data["uerr_prior_scale"] = 1
        self.stan_data["diff_prior_scale"] = 1
        self.stan_data["disc_prior_scale"] = 1
        self.stan_data = {**self.stan_data, **kwargs}
        self.opt = stan_model.optimizing(data=self.stan_data, verbose=True, iter=iter)
        print("sigma", self.opt["sigma"])
    def eval_model(self, random_scores, model_preds, modelname, num_samples):
        print(modelname)
        model_scores = eval_preds(model_preds, self.golddict, self.eval_fn)
        model_scores *= num_samples
        eval_scores_vs(random_scores, model_scores, self.badness_threshold)
        print("NUSERS", "NITEMS", "")

    def test(self, num_samples=5, debug=False):
        if self.simulator is None:
            raise ValueError("Must create simulator first")
        if self.opt is None:
            raise ValueError("Must train model first")
        per_item_user_rankings = get_model_user_rankings(self.opt, debug=debug)
        # TODO repeat random several times!
        random_scores = []
        for i in range(num_samples):
            random_preds = get_baseline_random(self.simulator.sim_df, self.label_colname, self.item_colname)
            random_scores += eval_preds(random_preds, self.golddict, self.eval_fn)
        self.faveu_preds = get_baseline_global_best_user(self.stan_data, self.simulator.sim_df, self.label_colname, self.item_colname, self.uid_name)
        self.center_preds = get_baseline_item_centrallest(self.stan_data, self.simulator.sim_df, self.label_colname, self.item_colname, self.uid_name)
        self.model_preds = get_preds(self.simulator.sim_df, per_item_user_rankings, self.label_colname, self.item_colname, self.uid_name)
        self.oracle_preds = get_oracle_preds(self.stan_data, self.simulator.sim_df, self.label_colname, self.item_colname, self.uid_name, self.eval_fn, self.golddict)
        self.eval_model(random_scores, self.faveu_preds, "FAVORITE USER", num_samples)
        self.eval_model(random_scores, self.center_preds, "SMALLEST DISTANCE", num_samples)
        self.eval_model(random_scores, self.model_preds, "MULTIVARIATE SCALING", num_samples)
        self.eval_model(random_scores, self.oracle_preds, "ORACLE", num_samples)
        if debug:
            params_model(self.simulator, self.opt, self.stan_data)
        
        # print("SMALLEST DISTANCE")
        # scores_sd = eval_preds(random, center, self.golddict, eval_fn=self.eval_fn, badness_threshold=self.badness_threshold, baseline_n_times=10)
        # print("MULTIVARIATE SCALING")
        # scores_ms = eval_preds(random, model4, self.golddict, eval_fn=self.eval_fn, badness_threshold=self.badness_threshold, baseline_n_times=10)
        # print("ORACLE")
        # scores_or = eval_preds(random, oracle, self.golddict, eval_fn=self.eval_fn, badness_threshold=self.badness_threshold, baseline_n_times=10)
    
class ParserExperiment(Experiment):
    def __init__(self):
        super().__init__("parse", "sentenceId")
        from nltk.data import find as nltkfind
        from nltk.parse.bllip import BllipParser
        bllip_dir = nltkfind('models/bllip_wsj_no_aux').path
        self.BLLIP = BllipParser.from_unified_model_dir(bllip_dir)
        self.eval_fn = evalb
        self.badness_threshold = 0.9

    def setup(self, num_items, n_users, pct_items, uerr_a, uerr_b, difficulty_a=1, difficulty_b=1,
                    ngoldu=0, min_sentence_length=10):
        from nltk.corpus import brown as browncorpus
        sentences = np.random.choice(browncorpus.sents(), num_items * 3, replace=False)
        sentences = [s for s in sentences if len(s) > min_sentence_length][:num_items]
        self.simulator = ParserSimulator(self.BLLIP, sentences)
        self.stan_data = self.simulator.create_stan_data_scenario(n_users=n_users, pct_items=pct_items,
                                                    uerr_a=uerr_a, uerr_b=uerr_b,
                                                    difficulty_a=difficulty_a, difficulty_b=difficulty_b,
                                                    n_gold_users=ngoldu)
        # TODO these are not true golds!
        gold_parses = [self.BLLIP.parse_one(ParsableStr(s)) for s in self.simulator.df.tokens.values]
        self.golddict = dict(enumerate(gold_parses))

def setup_parser_manyusers_hiuerr(parser_experiment):
    parser_experiment.setup(num_items=50, n_users=30, pct_items=0.2, uerr_a=2, uerr_b=2)
    
def setup_parser_fewusers_hiuerr(parser_experiment):
    parser_experiment.setup(num_items=50, n_users=10, pct_items=0.6, uerr_a=2, uerr_b=2)

def setup_parser_manyusers_hiuerr_fewitems(parser_experiment):
    parser_experiment.setup(num_items=25, n_users=30, pct_items=0.2, uerr_a=2, uerr_b=2)
    
def setup_parser_fewusers_hiuerr_fewitems(parser_experiment):
    parser_experiment.setup(num_items=25, n_users=10, pct_items=0.6, uerr_a=2, uerr_b=2)

def setup_parser_manyusers_hiuerr_somegold(parser_experiment):
    parser_experiment.setup(num_items=50, n_users=30, pct_items=0.2, uerr_a=2, uerr_b=2, ngoldu=3)

def setup_parser_manyusers_hiuerr_fewitems_somegold(parser_experiment):
    parser_experiment.setup(num_items=25, n_users=30, pct_items=0.2, uerr_a=2, uerr_b=2, ngoldu=3)

def setup_parser_huge(parser_experiment):
    parser_experiment.setup(num_items=150, n_users=30, pct_items=0.1, uerr_a=2, uerr_b=2)

class RankerExperiment(Experiment):
    def __init__(self, base_dir="qrels.all.txt"):
        super().__init__("rankings", "topic_item")
        self.base_dir = base_dir
        self.eval_fn = kendaltauscore
    def setup(self, n_items, n_users, pct_items, uerr_a, uerr_b, difficulty_a, difficulty_b, ngoldu=0):
        self.simulator = RankerSimulator(self.base_dir, n_items=n_items)
        self.stan_data = self.simulator.create_stan_data_scenario(n_users=n_users, pct_items=pct_items,
                                                    uerr_a=uerr_a, uerr_b=uerr_b,
                                                    difficulty_a=difficulty_a, difficulty_b=difficulty_b,
                                                    n_gold_users=ngoldu)
        self.golddict = self.simulator.gold.to_dict()
    def setup_standard(self):
        self.setup(n_items=100, n_users=20, pct_items=0.2, uerr_a=-1.0, uerr_b=0.3, difficulty_a=-2.0, difficulty_b=1.3, ngoldu=0)

def setup_ranker_manyusers_hiuerr_lodiff(ranker_experiment):
    ranker_experiment.setup(n_items=100, n_users=60, pct_items=0.2, uerr_a=-2, uerr_b=.5, difficulty_a=-3.0, difficulty_b=.1)
    
def setup_ranker_fewusers_hiuerr_lodiff(ranker_experiment):
    ranker_experiment.setup(n_items=100, n_users=20, pct_items=0.6, uerr_a=-2, uerr_b=.5, difficulty_a=-3.0, difficulty_b=.1)

def setup_ranker_manyusers_hiuerr_HIdiff(ranker_experiment):
    ranker_experiment.setup(n_items=100, n_users=60, pct_items=0.2, uerr_a=-2, uerr_b=.5, difficulty_a=-2.0, difficulty_b=.6)
    
def setup_ranker_fewusers_hiuerr_HIdiff(ranker_experiment):
    ranker_experiment.setup(n_items=100, n_users=20, pct_items=0.6, uerr_a=-2, uerr_b=.5, difficulty_a=-2.0, difficulty_b=.6)

def setup_ranker_manyusers_hiuerr_lodiff_somegold(ranker_experiment):
    ranker_experiment.setup(n_items=100, n_users=60, pct_items=0.2, uerr_a=-2, uerr_b=.5, difficulty_a=-3.0, difficulty_b=.1, ngoldu=6)

def setup_ranker_manyusers_hiuerr_HIdiff_somegold(ranker_experiment):
    ranker_experiment.setup(n_items=100, n_users=60, pct_items=0.2, uerr_a=-2, uerr_b=.5, difficulty_a=-2.0, difficulty_b=.6, ngoldu=6)


# def setup_ranker_manyusers_hiuerr_lodiff(ranker_experiment):
#     ranker_experiment.setup(n_items=100, n_users=60, pct_items=0.2, uerr_a=-1.0, uerr_b=0.6, difficulty_a=-2.0, difficulty_b=1.3)
    
# def setup_ranker_fewusers_hiuerr_lodiff(ranker_experiment):
#     ranker_experiment.setup(n_items=100, n_users=20, pct_items=0.6, uerr_a=-1.0, uerr_b=0.6, difficulty_a=-2.0, difficulty_b=1.3)

# def setup_ranker_manyusers_hiuerr_HIdiff(ranker_experiment):
#     ranker_experiment.setup(n_items=100, n_users=60, pct_items=0.2, uerr_a=-1.0, uerr_b=0.6, difficulty_a=-0.0, difficulty_b=1.6)
    
# def setup_ranker_fewusers_hiuerr_HIdiff(ranker_experiment):
#     ranker_experiment.setup(n_items=100, n_users=20, pct_items=0.6, uerr_a=-1.0, uerr_b=0.6, difficulty_a=-0.0, difficulty_b=1.6)

# def setup_ranker_manyusers_hiuerr_HIdiff_somegold(ranker_experiment):
#     ranker_experiment.setup(n_items=100, n_users=60, pct_items=0.2, uerr_a=-1.0, uerr_b=0.6, difficulty_a=-0.0, difficulty_b=1.6, ngoldu=6)

class VectorExperiment(Experiment):
    def __init__(self):
        super().__init__("label", "topic_item")
        self.eval_fn = lambda x, y: -euclidist(x, y)
    def setup(self, n_items, n_users, pct_items, uerr_a, uerr_b, difficulty_a, difficulty_b, ngoldu=0, n_dims=8):
        self.simulator = VectorSimulator(n_items, n_dims)
        self.stan_data = self.simulator.create_stan_data_scenario(n_users=n_users, pct_items=pct_items,
                                                    uerr_a=uerr_a, uerr_b=uerr_b, n_gold_users=ngoldu,
                                                    difficulty_a=difficulty_a, difficulty_b=difficulty_b)
        self.golddict = self.simulator.df.gold.to_dict()
def setup_vector_standard(vector_experiment):
    vector_experiment.setup(n_items=40, n_users=20, pct_items=0.2, uerr_a=-1.0, uerr_b=0.8,
                            difficulty_a=-2.0, difficulty_b=.3, ngoldu=10)
