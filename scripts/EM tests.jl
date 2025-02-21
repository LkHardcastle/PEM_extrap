using DrWatson
@quickactivate "PEM_extrap"
# For src
using DataStructures, LinearAlgebra, Distributions, Random, Optim, Roots, SpecialFunctions
using Plots, CSV, DataFrames, RCall, Interpolations

include(srcdir("Sampler.jl"))
include(srcdir("PreProcessing.jl"))
include(srcdir("PostProcessing.jl"))


R"""
library(ggplot2)
library(dplyr)
library(tidyr)
library(cowplot)
cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
"""

Random.seed!(12515)
n = 0
y = rand(Exponential(1.0),n)
breaks = collect(1:1:100)
p = 1
cens = fill(1.0,n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0))
nits = 50_000
nsmp = 100_000

Random.seed!(23462)
settings = Settings(nits, nsmp, 1_000_000, 10.0, 0.0, 5.0, false, true)
priors = EulerMaruyama(1.0, FixedV([0.01]), FixedW([0.5]), 1.0, Fixed(0.1), [GaussLangevin(2.0,0.2)])
@time out1 = pem_sample(state0, dat, priors, settings)
priors = BasicPrior(1.0, FixedV([0.01]), FixedW([0.5]), 1.0, Fixed(0.1), [GaussLangevin(2.0,0.2)])
@time out2 = pem_sample(state0, dat, priors, settings)
priors = EulerMaruyama(1.0, FixedV([0.05]), FixedW([0.5]), 1.0, Fixed(0.1), [GaussLangevin(2.0,0.2)])
@time out3 = pem_sample(state0, dat, priors, settings)
priors = BasicPrior(1.0, FixedV([0.05]), FixedW([0.5]), 1.0, Fixed(0.1), [GaussLangevin(2.0,0.2)])
@time out4 = pem_sample(state0, dat, priors, settings)
priors = EulerMaruyama(1.0, FixedV([0.1]), FixedW([0.5]), 1.0, Fixed(0.1), [GaussLangevin(2.0,0.2)])
@time out5 = pem_sample(state0, dat, priors, settings)
priors = BasicPrior(1.0, FixedV([0.1]), FixedW([0.5]), 1.0, Fixed(0.1), [GaussLangevin(2.0,0.2)])
@time out6 = pem_sample(state0, dat, priors, settings)
priors = EulerMaruyama(1.0, FixedV([0.25]), FixedW([0.5]), 1.0, Fixed(0.1), [GaussLangevin(2.0,0.2)])
@time out7 = pem_sample(state0, dat, priors, settings)
priors = BasicPrior(1.0, FixedV([0.25]), FixedW([0.5]), 1.0, Fixed(0.1), [GaussLangevin(2.0,0.2)])
@time out8 = pem_sample(state0, dat, priors, settings)
priors = EulerMaruyama(1.0, FixedV([0.4]), FixedW([0.5]), 1.0, Fixed(0.1), [GaussLangevin(2.0,0.2)])
@time out9 = pem_sample(state0, dat, priors, settings)
priors = BasicPrior(1.0, FixedV([0.4]), FixedW([0.5]), 1.0, Fixed(0.1), [GaussLangevin(2.0,0.2)])
@time out10 = pem_sample(state0, dat, priors, settings)

Random.seed!(23462)
settings = Settings(nits, nsmp, 1_000_000, 10.0, 2.0, 5.0, false, true)
priors = EulerMaruyama(1.0, FixedV([0.01]), FixedW([0.5]), 1.0, Fixed(0.1), [GaussLangevin(2.0,2.0)])
@time lout1 = pem_sample(state0, dat, priors, settings)
priors = BasicPrior(1.0, FixedV([0.01]), FixedW([0.5]), 1.0, Fixed(0.1), [GaussLangevin(2.0,2.0)])
@time lout2 = pem_sample(state0, dat, priors, settings)
priors = EulerMaruyama(1.0, FixedV([0.05]), FixedW([0.5]), 1.0, Fixed(0.1), [GaussLangevin(2.0,2.0)])
@time lout3 = pem_sample(state0, dat, priors, settings)
priors = BasicPrior(1.0, FixedV([0.05]), FixedW([0.5]), 1.0, Fixed(0.1), [GaussLangevin(2.0,2.0)])
@time lout4 = pem_sample(state0, dat, priors, settings)
priors = EulerMaruyama(1.0, FixedV([0.1]), FixedW([0.5]), 1.0, Fixed(0.1), [GaussLangevin(2.0,2.0)])
@time lout5 = pem_sample(state0, dat, priors, settings)
priors = BasicPrior(1.0, FixedV([0.1]), FixedW([0.5]), 1.0, Fixed(0.1), [GaussLangevin(2.0,2.0)])
@time lout6 = pem_sample(state0, dat, priors, settings)
priors = EulerMaruyama(1.0, FixedV([0.25]), FixedW([0.5]), 1.0, Fixed(0.1), [GaussLangevin(2.0,2.0)])
@time lout7 = pem_sample(state0, dat, priors, settings)
priors = BasicPrior(1.0, FixedV([0.25]), FixedW([0.5]), 1.0, Fixed(0.1), [GaussLangevin(2.0,2.0)])
@time lout8 = pem_sample(state0, dat, priors, settings)
priors = EulerMaruyama(1.0, FixedV([0.5]), FixedW([0.5]), 1.0, Fixed(0.1), [GaussLangevin(2.0,2.0)])
@time lout9 = pem_sample(state0, dat, priors, settings)
priors = BasicPrior(1.0, FixedV([0.5]), FixedW([0.5]), 1.0, Fixed(0.1), [GaussLangevin(2.0,2.0)])
@time lout10 = pem_sample(state0, dat, priors, settings)

s2 = cumsum(out1["Smp_x"], dims = 2)[1,:,:]
df2 = DataFrame(hcat(breaks, median(s2, dims = 2), quantile.(eachrow(s2), 0.025), quantile.(eachrow(s2), 0.25), quantile.(eachrow(s2), 0.75), quantile.(eachrow(s2), 0.975)), :auto)

R"""
dat2 = data.frame($df2)
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI") 
"""

R"""
p2 <- dat2 %>%
    pivot_longer(Mean:UCI) %>%
    ggplot(aes(x = Time, y = value, col = name, linetype = name)) + geom_step() +
    theme_classic() +
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,4,4,6)]) +
    scale_linetype_manual(values = c("dotdash","solid","dashed","dashed","dotdash")) + ylab("h(t)") + xlab("Time")
p2
#ggsave($plotsdir("GammaPrior.pdf"), width = 8, height = 6)
"""

df1 = out1["Smp_s"][1,:,:]
df2 = out2["Smp_s"][1,:,:]
df3 = out3["Smp_s"][1,:,:]
df4 = out4["Smp_s"][1,:,:]
df5 = out5["Smp_s"][1,:,:]
df6 = out6["Smp_s"][1,:,:]
df7 = out7["Smp_s"][1,:,:]
df8 = out8["Smp_s"][1,:,:]
df9 = out9["Smp_s"][1,:,:]
df10 = out10["Smp_s"][1,:,:]


df = DataFrame(EM1 = (sum(out1["Smp_s"][1,:,:], dims = 2)/length(out1["Smp_s"][1,5,:]))[2:end],
                Barker1 = (sum(out2["Smp_s"][1,:,:], dims = 2)/length(out2["Smp_s"][1,5,:]))[2:end],
                EM2 = (sum(out3["Smp_s"][1,:,:], dims = 2)/length(out3["Smp_s"][1,5,:]))[2:end],
                Barker2 = (sum(out4["Smp_s"][1,:,:], dims = 2)/length(out4["Smp_s"][1,5,:]))[2:end],
                EM3 = (sum(out5["Smp_s"][1,:,:], dims = 2)/length(out5["Smp_s"][1,5,:]))[2:end],
                Barker3 = (sum(out6["Smp_s"][1,:,:], dims = 2)/length(out6["Smp_s"][1,5,:]))[2:end],
                EM4 = (sum(out7["Smp_s"][1,:,:], dims = 2)/length(out7["Smp_s"][1,5,:]))[2:end],
                Barker4 = (sum(out8["Smp_s"][1,:,:], dims = 2)/length(out8["Smp_s"][1,5,:]))[2:end],
                EM5 = (sum(out9["Smp_s"][1,:,:], dims = 2)/length(out9["Smp_s"][1,5,:]))[2:end],
                Barker5 = (sum(out10["Smp_s"][1,:,:], dims = 2)/length(out10["Smp_s"][1,5,:]))[2:end])
CSV.write(datadir("EM_exp1.csv"), df)
R"""
$df %>%
    pivot_longer(c(EM1:Barker5)) %>%
    mutate(method = case_when(
        grepl("EM", name, fixed = TRUE) ~ "EM",
        grepl("Barker", name, fixed = TRUE) ~ "Barker"
            ),
            step_size = case_when(
                grepl("1", name, fixed = TRUE) ~ "0.01",
                grepl("2", name, fixed = TRUE) ~ "0.05",
                grepl("3", name, fixed = TRUE) ~ "0.1",
                grepl("4", name, fixed = TRUE) ~ "0.25",
                grepl("5", name, fixed = TRUE) ~ "0.5"
            )) %>%
    ggplot(aes(x = step_size, y = value, col = method)) + geom_boxplot() +
    theme_classic() + scale_colour_manual(values = cbPalette[6:7]) + geom_hline(yintercept = 0.5, linetype = "dotted")
"""

df = DataFrame(EM1 = (sum(lout1["Smp_s"][1,:,:], dims = 2)/length(lout1["Smp_s"][1,5,:]))[2:end],
                Barker1 = (sum(lout2["Smp_s"][1,:,:], dims = 2)/length(lout2["Smp_s"][1,5,:]))[2:end],
                EM2 = (sum(lout3["Smp_s"][1,:,:], dims = 2)/length(lout3["Smp_s"][1,5,:]))[2:end],
                Barker2 = (sum(lout4["Smp_s"][1,:,:], dims = 2)/length(lout4["Smp_s"][1,5,:]))[2:end],
                EM3 = (sum(lout5["Smp_s"][1,:,:], dims = 2)/length(lout5["Smp_s"][1,5,:]))[2:end],
                Barker3 = (sum(lout6["Smp_s"][1,:,:], dims = 2)/length(lout6["Smp_s"][1,5,:]))[2:end],
                EM4 = (sum(lout7["Smp_s"][1,:,:], dims = 2)/length(lout7["Smp_s"][1,5,:]))[2:end],
                Barker4 = (sum(lout8["Smp_s"][1,:,:], dims = 2)/length(lout8["Smp_s"][1,5,:]))[2:end],
                EM5 = (sum(lout9["Smp_s"][1,:,:], dims = 2)/length(lout9["Smp_s"][1,5,:]))[2:end],
                Barker5 = (sum(lout10["Smp_s"][1,:,:], dims = 2)/length(lout10["Smp_s"][1,5,:]))[2:end])
CSV.write(datadir("EM_exp2.csv"), df)
R"""
$df %>%
    pivot_longer(c(EM1:Barker5)) %>%
    mutate(method = case_when(
        grepl("EM", name, fixed = TRUE) ~ "EM",
        grepl("Barker", name, fixed = TRUE) ~ "Barker"
            ),
            step_size = case_when(
                grepl("1", name, fixed = TRUE) ~ "0.01",
                grepl("2", name, fixed = TRUE) ~ "0.05",
                grepl("3", name, fixed = TRUE) ~ "0.1",
                grepl("4", name, fixed = TRUE) ~ "0.25",
                grepl("5", name, fixed = TRUE) ~ "0.5"
            )) %>%
    ggplot(aes(x = step_size, y = value, col = method)) + geom_boxplot() +
    theme_classic() + scale_colour_manual(values = cbPalette[6:7]) + geom_hline(yintercept = 0.5, linetype = "dotted")
"""
