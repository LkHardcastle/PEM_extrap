#
setwd("C:\\Users\\hardc\\Documents\\PEM_extrap\\Scripts\\Colon")
library(dplyr)
library(tidyverse)
library(rstan)
library(ggplot2)
library(zoo)
#install.packages("survextrap", repos=c('https://chjackson.r-universe.dev',
#                                       'https://cloud.r-project.org'))
library(survextrap)
library(BayesReversePLLH)
# Survextrap


nd_modr1 <- survextrap(Surv(years, status) ~ 1, data=colons, chains=2, 
                      smooth_model = "random_walk",
                      mspline = list(add_knots=4))
spline_out1 = hazard(nd_modr1, t = seq(0.01, 15, .01))
plot(nd_modr1)
extdat <- data.frame(start = c(10), stop =  c(15), 
                     n = c(20), r = c(6))
nd_modr2 <- survextrap(Surv(years, status) ~ 1, data=colons, chains=2, 
                      smooth_model = "random_walk", external = extdat)

spline_out2 = hazard(nd_modr2, t = seq(0.01, 15, .01))

# Cooney

#devtools::install_github("Anon19820/PiecewiseChangepoint")
library("PiecewiseChangepoint")

n_obs =300
n_events_req=300
max_time =  24 # months

rate = c(0.75,0.25)/12 # we want to report on months
t_change =12 # change-point at 12 months

df <- gen_piece_df(n_obs = n_obs,n_events_req = n_events_req,
                   num.breaks = length(t_change),rate = rate ,
                   t_change = t_change, max_time = max_time)

Collapsing_Model <- collapsing.model(data.frame(status = colons$status, time = colons$years),
                                     n.iter = 20750,
                                     burn_in = 750,
                                     n.chains = 2,
                                     timescale = "years")
print(Collapsing_Model)
plot(Collapsing_Model, type = "hazard")+xlab("Time (Months)")+ylab("Hazards")+ylim(c(0,1))
plot(Collapsing_Model, max_predict = 5)+xlab("Time")

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


# Kearns

library("KFAS")
library("discSurv")

colons <- arrange(colons, years)
colons$eventtime <- as.integer(colons$years*12) + 1
ltcolons <- lifeTable(as.data.frame(colons), timeColumn = "eventtime", eventColumn = "status")
ltHaz <- data.frame(hazKM = ltcolons$Output$hazard, Time = (seq(1:length(ltcolons$Output[,1]))-0.5)/12,
                    AtRisk = ltcolons$Output$atRisk, Events = ltcolons$Output$events)
# Final month
follow_up <- max(colons$eventtime)
# The above hazard is the product-limit (KM) estimate. Also calculate the life-table (acturial) estimate
ltHaz$hazLT = ltHaz$Events / (ltHaz$AtRisk - ltHaz$Events/2)
# Generate log-time
ltHaz$lnTime <- log(ltHaz$Time)
# For random effects add an ID for each time period
ltHaz$MyId <- 1:dim(ltHaz)[1] # Generate id variable 
# For AR(1) model get outcomes lagged by one.
ltHaz$EventsL <- lag(ltHaz$Events)
# Set first lagged value = 0 (usually would discard, but retain so IC are comparable. Can be justified as a prior value)
ltHaz$EventsL[1] <- 0

logTime <- data.frame(matrix(nrow=dim(ltHaz)[1], ncol=4))
colnames(logTime) <- c("Time","Events","AtRisk","Hazard")
# First get equi-spaced in log-time
MyMin <- log(min(ltHaz$Time))
MyMax <- log(max(ltHaz$Time))
MyNum <- follow_up # Number of points we want
logTime$Time <- MyMin+(MyMax-MyMin)/MyNum*(index(logTime)-1)
# Now exponentiate
logTime$Time <- exp(logTime$Time)
logTime$Events <- round(approx(ltHaz$Time,ltHaz$Events, xout=logTime$Time)$y, digits=0)
logTime$AtRisk <- approx(ltHaz$Time,ltHaz$AtRisk, xout=logTime$Time)$y
logTime$Hazard <- (logTime$Events / logTime$AtRisk)

strcLvl <- SSModel(logTime$Events ~ -1 + 
                     SSMtrend(degree = 1, Q = list(matrix(NA))),
                   distribution="poisson", u=logTime$AtRisk)
strcTrnd <- SSModel(logTime$Events ~ -1 + 
                      SSMtrend(degree = 2, Q = list(matrix(NA), matrix(NA))),
                    distribution="poisson", u=logTime$AtRisk)
strcDrft <- SSModel(logTime$Events ~ -1 + 
                      SSMtrend(degree = 2, Q = list(matrix(NA), matrix(0))),
                    distribution="poisson", u=logTime$AtRisk)
# Now fit and get estimates (N.B. for replication using nsim = 0, that is Gaussian approximation only)
modLvl <- fitSSM(strcLvl, c(0.1), method = "BFGS")$model
estLvl <- KFS(modLvl, nsim = 0)
modTrnd <- fitSSM(strcTrnd, c(0.1,0.1), method = "BFGS")$model
estTrnd <- KFS(modTrnd, nsim = 0)
modDrft <- fitSSM(strcDrft, c(0.1,0.1), method = "BFGS")$model
estDrft <- KFS(modDrft, nsim = 0) 

predLvl <- data.frame(predict(object=modLvl,
                              newdata = SSModel(ts(matrix(NA, 50*12-follow_up, 1), start = 1) ~ -1 +
                                                  SSMtrend(degree = 1, Q = list(matrix(modLvl$Q[1]))),
                                                distribution="poisson", u=1),
                              type = "response", interval = "confidence", nsim = 0))
predTrnd <- data.frame(predict(object=modTrnd,
                               newdata = SSModel(ts(matrix(NA, 50*12-follow_up, 1), start = 1) ~ -1 +
                                                   SSMtrend(degree = 2, Q = list(matrix(modTrnd$Q[1]), matrix(modTrnd$Q[4]))),
                                                 distribution="poisson", u=1),
                               type = "response", interval = "confidence", nsim = 0))
predDrft <- data.frame(predict(object=modDrft,
                               newdata = SSModel(ts(matrix(NA, 50*12-follow_up, 1), start = 1) ~ -1 +
                                                   SSMtrend(degree = 2, Q = list(matrix(modDrft$Q[1]), matrix(modDrft$Q[4]))),
                                                 distribution="poisson", u=1),
                               type = "response", interval = "confidence", nsim = 0))

# Generate hazard for observed period - again in log-time
estDLM <- data.frame(Level = estLvl$muhat/logTime$AtRisk, Trend = estTrnd$muhat/logTime$AtRisk,
                     Drift = estDrft$muhat/logTime$AtRisk, logTime = log(logTime$Time), Time = logTime$Time)

step <- estDLM$logTime[2] - estDLM$logTime[1]
# Collate predictions
pred <- data.frame(Level = predLvl$fit, Trend = predTrnd$fit, Drift = predDrft$fit)
# Include time period & change to usual scale
pred$logTime <- estDLM$logTime[dim(estDLM)[1]] + index(pred)*step
pred$Time <- exp(pred$logTime)
# Keep observations for times < 50
pred2 <- subset(pred, Time < 50)
# Append observed and predicted times
dfDGLM <- rbind(estDLM, pred2)
ltHaz$Level <- ltHaz$AtRisk * approx(x=dfDGLM$Time, y=dfDGLM$Level, xout=ltHaz$Time)$y
ltHaz$Trend <- ltHaz$AtRisk * approx(x=dfDGLM$Time, y=dfDGLM$Trend, xout=ltHaz$Time)$y
ltHaz$Drift <- ltHaz$AtRisk * approx(x=dfDGLM$Time, y=dfDGLM$Drift, xout=ltHaz$Time)$y
dfDGLM = dfDGLM %>% select(-logTime) %>% gather(key = "Model", value = "Haz", -Time)

DSM_out <- dfDGLM

# Chapple

out <- BayesPiecewiseHazard(colons$years, colons$status, 100, 10000)
# Split points
out[[1]]
# log-hazards
plot(out[[2]][,1], type = "l")
# No. split points
plot(out[[3]])
# Variance
out[[4]]
# RMST
out[[5]]
plot(out)

n <- 100
int_prob <- runif(n)
int <- ifelse(int_prob < 1-exp(-1), 1, ifelse(int_prob < 1-exp(-1)*exp(-1.1),2,3))
cens <- ifelse(int == 3, 0,1)

y <- rep(NA,n)
for(i in 1:100){
  y[i] <- ifelse(int[i] == 1, runif(1,min = 0, max = 1)[1], ifelse(int[i] == 2, runif(1,min = 1,max = 2)[1], 2))
}

out2 <- BayesPiecewiseHazard(y, cens, 5, 1000000)

plot(out2[[3]])

# Save 
write.csv(spline_out1, "C:\\Users\\hardc\\Documents\\PEM_extrap\\data\\ColonSmps\\spline.csv")
write.csv(spline_out2, "C:\\Users\\hardc\\Documents\\PEM_extrap\\data\\ColonSmps\\spline_ext.csv")
write.csv(DSM_out, "C:\\Users\\hardc\\Documents\\PEM_extrap\\data\\ColonSmps\\DSM.csv")
write.csv(pw_out, "C:\\Users\\hardc\\Documents\\PEM_extrap\\data\\ColonSmps\\PW.csv")




