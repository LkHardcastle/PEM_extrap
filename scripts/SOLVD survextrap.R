setwd("C:\\Users\\hardc\\Documents\\PEM_extrap")
library(dplyr)
library(tidyverse)
library(rstan)
library(ggplot2)
library(zoo)
#install.packages("survextrap", repos=c('https://chjackson.r-universe.dev',
#                                       'https://cloud.r-project.org'))
library(survextrap)


dat <- read.csv("C:\\Users\\hardc\\Documents\\PEM_extrap\\data\\SOLVD\\SOLVD.csv")
dat <- dat %>%
  subset(EPYTIME > 0) %>%
  subset(TRIAL == "P")

dat$cens <-  as.numeric(nchar(dat$DDATE) > 0)
dat$years = dat$FUTIME/365
# Survextrap
nd_modr1 <- survextrap(Surv(years,cens) ~ DRUG, data=dat, chains=2, 
                       smooth_model = "random_walk",
                       mspline = list(add_knots=6))

plot(nd_modr1)

nd_modr2 <- survextrap(Surv(years,cens) ~ DRUG, data=dat, chains=2, 
                       smooth_model = "random_walk", nonprop =TRUE,
                       mspline = list(add_knots=6))

plot(nd_modr2, tmax = 15, line_size = 0.7, ci = TRUE)
