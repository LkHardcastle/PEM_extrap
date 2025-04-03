#
setwd("C:\\Users\\hardc\\Documents\\PEM_extrap\\Scripts\\Colon")
library(dplyr)
library(tidyverse)
library(rstan)
library(ggplot2)
library(zoo)
library(survextrap)
library(BayesReversePLLH)
library("PiecewiseChangepoint")
library("KFAS")
library("discSurv")

# Spline models

nd_modr1 <- survextrap(Surv(years, status) ~ 1, data=colons, chains=2, 
                      smooth_model = "random_walk",
                      mspline = list(add_knots=5))
plot(nd_modr1)
nd_modr2 <- survextrap(Surv(years, status) ~ 1, data=colons, chains=2, 
                      smooth_model = "random_walk",
                      mspline = list(add_knots=10))
plot(nd_modr2)
nd_modr3 <- survextrap(Surv(years, status) ~ 1, data=colons, chains=2, 
                      smooth_model = "random_walk",
                      mspline = list(add_knots=15))
plot(nd_modr3)

spline_out1 = hazard(nd_modr1, t = seq(0.01, 15, .001))



extdat <- data.frame(start = c(10), stop =  c(15), 
                     n = c(20), r = c(6))
nd_modr2 <- survextrap(Surv(years, status) ~ 1, data=colons, chains=2, 
                      smooth_model = "random_walk", external = extdat)
spline_out2 = hazard(nd_modr2, t = seq(0.01, 15, .001))

rmst(nd_modr1, t = c(3.1, 15), summ_fns = list(mean = mean, ~quantile(.x, probs=c(0.025, 0.975))), niter=1000)
rmst(nd_modr2, t = c(3.1, 15), summ_fns = list(mean = mean, ~quantile(.x, probs=c(0.025, 0.975))),niter=1000)
rmst(nd_modr3, t = c(3.1, 15), summ_fns = list(mean = mean, ~quantile(.x, probs=c(0.025, 0.975))),niter=1000)


rmst(nd_modr2, t = c(3.1,15), niter=100)
write.csv(spline_out1, "C:\\Users\\hardc\\Documents\\PEM_extrap\\data\\ColonSmps\\spline.csv")
write.csv(spline_out2, "C:\\Users\\hardc\\Documents\\PEM_extrap\\data\\ColonSmps\\spline_ext.csv")

# Piecewise exponential (independent)

library("PiecewiseChangepoint")

Collapsing_Model <- collapsing.model(data.frame(status = colons$status, time = colons$years),
                                     n.iter = 20000,
                                     burn_in = 5000,
                                     n.chains = 2,
                                     timescale = "years")


object = Collapsing_Model
chng.num = "all"
max.num.post = 500
alpha.pos = NULL
k <- object$k.stacked
changepoint <- object$changepoint
lambda <- object$lambda
df <- object$df
interval <- max(df$time)/100
time.seq <- c(seq(from = 0, to = max(df$time), by = interval))
num.changepoints <- unlist(apply(k, 1, function(x) {
  length(na.omit(x))
}))
if (chng.num != "all") {
  lambda <- as.matrix(lambda[which(num.changepoints == 
                                     chng.num), 1:(chng.num + 1)])
  changepoint <- as.matrix(changepoint[which(num.changepoints == 
                                               chng.num), 1:chng.num])
  num.changepoints <- num.changepoints[which(num.changepoints == 
                                               chng.num)]
}
lambda_res_final <- NULL
for (i in seq_along(unique(num.changepoints))) {
  index <- unique(num.changepoints)[order(unique(num.changepoints))][i]
  if (index == 0) {
    lambda_curr <- lambda[which(num.changepoints == index), 
                          1:(index + 1)]
    lambda_res_final <- matrix(rep(lambda_curr, each = length(time.seq)), 
                               nrow = length(lambda_curr), ncol = length(time.seq), 
                               byrow = T)
    df.changepoint <- data.frame(timepoints = rep(c(0, 
                                                    max(df$time)), times = length(lambda_curr)), 
                                 hazards = rep(lambda_curr, each = 2), id = rep(1:length(lambda_curr), 
                                                                                each = 2))
  }
  else {
    changepoint_curr <- as.matrix(changepoint[which(num.changepoints == 
                                                      index), 1:index])
    lambda_curr <- lambda[which(num.changepoints == index), 
                          1:(index + 1)]
    lambda_res_curr <- matrix(nrow = nrow(changepoint_curr), 
                              ncol = length(time.seq))
    changepoint_curr_samp <- cbind(changepoint_curr, 
                                   Inf)
    for (j in 1:length(time.seq)) {
      index.lambda <- apply(changepoint_curr_samp, 
                            1, function(x) {
                              which.min(time.seq[j] > x)
                            })
      lambda_res_curr[, j] <- lambda_curr[cbind(1:nrow(changepoint_curr_samp), 
                                                index.lambda)]
    }
    lambda_res_final <- rbind(lambda_res_final, lambda_res_curr)
  }
}
df.hazard1 <- data.frame(timepoints = rep(time.seq, by = nrow(lambda_res_final)), 
                        hazards = c(t(lambda_res_final)), id = rep(1:nrow(lambda_res_final), 
                                                                   each = length(time.seq)))
pw_out <- df.hazard1 %>%
  group_by(timepoints) %>%
  summarise(mean = mean(hazards),
            UCI = quantile(hazards, 0.975),
            LCI = quantile(hazards, 0.025))
St_all <- get_Surv(Collapsing_Model, time = seq(0.01, 15, by = 0.01))
meanSt_obs <- colSums(St_all[1:310,]*0.01)
meanSt_all <- colSums(St_all*0.01)
mean(meanSt_all)
mean(meanSt_obs)
quantile(meanSt_obs, c(0.025,0.975))
quantile(meanSt_all, c(0.025,0.975))

write.csv(pw_out, "C:\\Users\\hardc\\Documents\\PEM_extrap\\data\\ColonSmps\\PW.csv")

##### DSMs

# Define time cuts

times = sort(unique(colons$years))
knots = times[seq(8, length(times), 8)]
t_plot = times[seq(8, length(times), 8)]
T_ = length(knots)

n_ = data.frame(bins = cut(colons$years, c(0,knots))) %>% group_by(bins) %>% summarize(count=n())
y_ = data.frame(bins = cut(colons$years[colons$status == 1], c(0,knots))) %>% group_by(bins) %>% summarize(count=n())
knots = c(knots, 3.1)*12

my_data <- list(
  T = length(knots),
  tau = knots[2:length(knots)] - knots[1:(length(knots)-1)] ,
  y = y_$count,
  n = rev(cumsum(n_$count))
)
mod1 <- stan(
    file = "DSM_damped.stan",    # Stan program - change when fitting different models
    data = my_data,         # named list of data
    chains = 2
  )

mod2 <- stan(
    file = "DSM_local.stan",    # Stan program - change when fitting different models
    data = my_data,         # named list of data
    chains = 2
  )

plot(mod1)
plot(mod2)

 beta1 = extract(mod1, pars = c("beta1"))
  int_haz = map_dfr(beta1, function(x) colMeans(x))
  tmp2 = tibble(Time = df$Time, mean = int_haz$beta1)
    max_fu = max(tmp2$Time) # = tmp2$Time[length(tmp2$Time)] as ordered
  time_int = filter(new_df, Time <= max_fu) 
    int_est = approx(x=tmp2$Time, y=tmp2$mean, xout=time_int$Time, rule=2)$y


### SurvHE
library(survHE)
out = fit.models(formula=Surv(years,status)~1,data=colons,
                distr=c("exponential","gamma","gompertz","weibull","loglogistic","lognormal"),method="mle")

out$model.fitting
out$models$'log-Normal'

summary.survHE(out, t = seq(0.1,15,by= 0.1))
