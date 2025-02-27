
data {
    int<lower=1> T;  // Time points
    vector[T-1] tau;   // Width between time-points
    int y[T];        // Events
    vector[T] n;     // At risk
}

parameters {
    real beta_01;              // Initial coeff1
    real<lower=0> Z;          // Variance coeff2
    real beta_02;              // Initial coeff2
    vector[T-1] zeta_tilde;      // Tranformation of zeta2 (as in 8-schools example)
}

transformed parameters {
    vector[T] beta1;         // State 1
    vector[T-1] beta2;         // State 2
    { // Don't want to save this
      vector[T-1] zeta2;         // Innovations

      zeta2 = sqrt(Z) * zeta_tilde;
      beta1[1] = beta_01;
      beta2[1] = beta_02 + zeta2[1];
      for (t in 2:T-1) {
        beta1[t] = beta1[t-1] + beta2[t-1] * tau[t-1];
        beta2[t] = beta2[t-1] + zeta2[t];
      }
      beta1[T] = beta1[T-1] + beta2[T-1] * tau[T-1];
    }
}

model {
  Z ~ inv_gamma(1, 0.005);
  zeta_tilde ~ normal(0, 1);
  y ~ poisson(exp(beta1) .* n);
}

generated quantities{
  real level;
  real trend;

  level = beta1[T];
  trend = beta2[T-1];
}
