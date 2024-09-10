#
setwd("C:\\Users\\hardc\\Documents\\PEM_extrap\\Scripts\\Colon")
library(dplyr)
library(rstan)
library(ggplot2)
install.packages("survextrap", repos=c('https://chjackson.r-universe.dev',
                                       'https://cloud.r-project.org'))
library(survextrap)

# Survextrap
nd_modr <- survextrap(Surv(years, status) ~ 1, data=colons, chains=2, 
                      smooth_model = "random_walk",
                      mspline = list(add_knots=4))
pairs(nd_modr$stanfit, pars = c("coefs"))
plot(nd_modr, tmax=5)

# Cooney

devtools::install_github("Anon19820/PiecewiseChangepoint")
library(PiecewiseChangepoint)

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


# Kearns

library("KFAS")
