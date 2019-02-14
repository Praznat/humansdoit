data {
    # observed data
    int<lower=0> NDATA;
    int<lower=0> NITEMS;
    int<lower=0> NUSERS;
    int<lower=1, upper=NITEMS> items[NDATA];
    int<lower=1, upper=NUSERS> u1s[NDATA];
    int<lower=1, upper=NUSERS> u2s[NDATA];
    real similarities[NDATA];

    # hyperparameters
    real c_prior_mu;
    real<lower=0> c_prior_scale;
}
parameters {
    vector<lower=0>[NUSERS] err;
    vector<lower=0>[NITEMS] diff;
    vector<lower=0, upper=1>[NITEMS] amb;
    real c1_Z;
    real c2_Z;
    real ca_Z;
    real cd_Z;
}
transformed parameters {
    real c1 = c_prior_mu + c_prior_scale * c1_Z;
    real c2 = c_prior_mu + c_prior_scale * c2_Z;
    real ca = c_prior_mu + c_prior_scale * ca_Z;
    real cd = c_prior_mu + c_prior_scale * cd_Z;
    vector<lower=0, upper=1>[NITEMS] unamb = 1 - amb;
}
model {
    err ~ normal(0, 1);
    diff ~ normal(0, 1);
    c1_Z ~ normal(0, 1);
    c2_Z ~ normal(0, 1);
    ca_Z ~ normal(0, 1);
    cd_Z ~ normal(0, 1);
    for (i in 1:NDATA) {
        int u1 = u1s[i];
        int u2 = u2s[i];
        int item = items[i];
        real sig2 = unamb[item]^ca * (err[u1]^c1 + err[u2]^c2) + 2 * amb[item]^2 * diff[item]^2;
        similarities[i] ~ normal(0, sqrt(sig2));
    }
}