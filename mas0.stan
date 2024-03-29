functions {
    // no built-in vector norm in Stan?
    real norm(vector x) {
        return sqrt(sum(square(x)));
    }
}

data {
    # observed data
    int<lower=0> NDATA;
    int<lower=0> NITEMS;
    int<lower=0> NUSERS;
    int<lower=1, upper=NITEMS> items[NDATA];
    int<lower=1, upper=NUSERS> u1s[NDATA];
    int<lower=1, upper=NUSERS> u2s[NDATA];
    int<lower=0, upper=NUSERS> n_gold_users; # first n_gold_users assumed to be gold
    real gold_user_err;
    real distances[NDATA];

    # hyperparameters
    int<lower=1> DIM_SIZE;
    int<lower=0> eps_limit;
    int<lower=0, upper=1> use_diff;
    int<lower=0, upper=1> use_disc;
    int<lower=0, upper=1> use_norm;
    real<lower=0, upper=1> norm_ratio;
    real<lower=0> uerr_prior_scale;
    real<lower=0> diff_prior_scale;
    real<lower=0> disc_prior_scale;

    real<lower=0> uerr_prior_loc_scale;
    real<lower=0> diff_prior_loc_scale;
    real<lower=0> disc_prior_loc_scale;
}
transformed data {
    real scaling = 1.0 / DIM_SIZE;
}
parameters {
    real uerr_prior_loc;
    real diff_prior_loc;
    real disc_prior_loc;
    vector[use_diff ? NITEMS : 0] diff_Z;
    vector[use_disc ? NITEMS : 0] disc_Z;
    vector[NUSERS] uerr_Z;
    real<lower=0> sigma;

    vector<lower=-eps_limit, upper=eps_limit>[DIM_SIZE] item_user_errors_Z[NITEMS, NUSERS];
}
transformed parameters {
    real pred_distances[NDATA];
    vector[DIM_SIZE] item_user_errors[NITEMS, NUSERS];

    vector<lower=0>[use_diff ? NITEMS : 0] diff = exp(diff_prior_loc + diff_Z * diff_prior_scale);
    vector<lower=0>[use_disc ? NITEMS : 0] disc = exp(disc_prior_loc + disc_Z * disc_prior_scale);
    vector<lower=0>[NUSERS] uerr = exp(uerr_prior_loc + uerr_Z * uerr_prior_scale);

    matrix<lower=0>[NITEMS, NUSERS] dist_from_truth = rep_matrix(666, NITEMS, NUSERS);

    // for (u in 1:n_gold_users) uerr[u] = gold_user_err;
    for (u in 1:n_gold_users) uerr[u] = exp(uerr_prior_loc + gold_user_err * uerr_prior_scale);
    for (i in 1:NITEMS) {
        for (u in 1:NUSERS) {
            vector[DIM_SIZE] epsilon;
            real err = uerr[u];
            if (use_diff) err += diff[i];
            if (use_disc) err *= disc[i];
            epsilon = err * item_user_errors_Z[i, u];
            if (use_norm) {
                epsilon = epsilon * (1 - norm_ratio) + epsilon / norm(item_user_errors_Z[i, u]) * norm_ratio;
            }
            item_user_errors[i, u] = epsilon;
        }
    }

    for (n in 1:NDATA) {
        int u1 = u1s[n];
        int u2 = u2s[n];
        int item = items[n];
        dist_from_truth[item, u1] = norm(item_user_errors[item, u1]);
        dist_from_truth[item, u2] = norm(item_user_errors[item, u2]);
        pred_distances[n] = norm(item_user_errors[item, u1] - item_user_errors[item, u2]);
    }
}
model {
    // sigma ~ exponential(1);
    uerr_Z ~ normal(0, 1);
    diff_Z ~ normal(0, 1);
    disc_Z ~ normal(0, 1);
    uerr_prior_loc ~ normal(0, uerr_prior_loc_scale);
    diff_prior_loc ~ normal(0, diff_prior_loc_scale);
    disc_prior_loc ~ normal(0, disc_prior_loc_scale);
    // for (i in 1:NITEMS) {
    //     for (u in 1:NUSERS) {
    //         item_user_errors_Z[i, u] ~ normal(0, 1);
    //     }
    // }

    distances ~ normal(pred_distances, sigma);
}
