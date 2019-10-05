import numpy as np
from tqdm import tqdm
from itertools import combinations
import matplotlib.pyplot as plt
import pandas as pd
import pystan
import networkx as nx

# Load data

golds = pd.read_csv("gold.txt", sep=" ", header=None, names=["item", "gold"])
if golds.gold.max() > 1:
    golds.gold = (golds.gold/2).astype(int)

orig_df = pd.read_csv("responses.txt", sep=" ", header=None, names=["user", "item", "label"])
if orig_df.label.max() > 1:
    orig_df.label = (orig_df.label/2).astype(int)

uc = orig_df.groupby("item").count()["user"]
baditems = uc[uc<2].keys()
orig_df = orig_df[~orig_df.item.isin(baditems)]

def add_spammer(df, new_uid=None, random_numitems=True):
    if new_uid is None:
        new_uid = df.user.unique().max() + 1
    ucounts = df.groupby("user").count()["item"]
    if random_numitems:
        numitemsdone = np.random.choice(ucounts, 1)[0]
    else:
        numitemsdone = ucounts.median()
    itemsdone = np.random.choice(df.item.unique(), numitemsdone)
    p = 0.5
    labels = np.random.binomial(1, p, len(itemsdone))
    newdf = pd.DataFrame({"user":new_uid, "item":itemsdone, "label":labels})
    return newdf

def create_spammer_df(base_df, num_spammers, random_numitems=True):
    result_df = []
    for i in range(num_spammers):
        new_uid = "spammy" + str(i)#df.user.unique().max() + 1 + i
        result_df.append(add_spammer(base_df, new_uid=new_uid, random_numitems=random_numitems))
    return pd.concat(result_df)
        
# Add spammers

nspammers = int(len(orig_df.user.unique()) * .25)
spam_df = create_spammer_df(orig_df, nspammers)

# Add gold

df = pd.concat([orig_df, spam_df])
df = df.merge(golds, how="left", on="item")
df.user = pd.Categorical(df.user).codes
df.item = pd.Categorical(df.item).codes

agree_fn = lambda x, y: -np.abs(x - y)

def get_1round_agreement(df, samplesize=100):
    idf = df.groupby("item").mean()["label"]
    skills = np.zeros(len(df.user.unique()))
    for u in tqdm(df.user.unique()):
        udf = df[df.user==u]
        udf = udf.sample(n=min(len(udf), samplesize))
        agreements = []
        for i, row in udf.iterrows():
            avglabel = idf.loc[row["item"]]
            agreements.append(agree_fn(row.label, avglabel))
        skills[u] = np.mean(agreements)
    return skills

baselineskill = get_1round_agreement(df)
baseskillbins = np.digitize(baselineskill, bins=np.nanquantile(baselineskill, q=np.linspace(0,1,100)))

# Similarity matrix

n_users = len(df.user.unique())
n_items = len(df.item.unique())
numer = np.zeros((n_users, n_users))
denom = np.zeros((n_users, n_users))

for item in tqdm(df.item.unique()):
    idf = df[df.item == item]
    for users, labels in zip(combinations(idf.user, 2), combinations(idf.label, 2)):
        u0, u1 = users
        agreement = agree_fn(labels[0], labels[1])
        numer[u0, u1] += agreement
        denom[u0, u1] += 1

w_mat = 1 + (numer / denom)

# Page-rank?

iters = 25
alpha = 0.1
beta = 0.6
beta_v = beta * np.ones(w_mat.shape[0])
skill = beta * np.ones(w_mat.shape[0])
for i in range(iters):
    xx = np.nanmean(skill * w_mat, axis=0)
    delta = np.nansum((xx - skill)**2)
    skill = alpha * xx + (1 - alpha) * beta

# Diagnostics

xx = np.nanmean(w_mat, axis=0)
plt.hist(xx)
plt.show()
plt.hist(skill)
plt.show()

propskillbins = np.digitize(skill, bins=np.nanquantile(skill, q=np.linspace(0,1,100)))

G = nx.from_numpy_matrix(np.nan_to_num(w_mat,0))
pos = nx.spring_layout(G)
nx.draw(G, pos, node_size=50, node_color=propskillbins, cmap=plt.cm.RdYlGn, alpha=0.6)
plt.show()

# Evaluation

def accuracy(pred, actual):
    rpred = np.round(pred)
    return np.mean(rpred * actual + (1 - rpred) * (1 - actual))

goldlabels = df.groupby("item").mean()["gold"].values

majorityvote = df.groupby("item").mean()["label"]
print(accuracy(majorityvote, goldlabels))

culled_df = df[df.user.isin(np.where(baseskillbins>40)[0])]
culled_gold = culled_df.groupby("item").mean()["gold"].values
culled_mv = culled_df.groupby("item").mean()["label"]
print(accuracy(culled_mv, culled_gold))

propculled_df = df[df.user.isin(np.where(skillbins>40)[0])]
propculled_gold = propculled_df.groupby("item").mean()["gold"].values
propculled_mv = propculled_df.groupby("item").mean()["label"]
print(accuracy(propculled_mv, propculled_gold))