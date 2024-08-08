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
breaks = collect(1:1:200)
p = 1
cens = fill(1.0,n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0), ones(size(x0)))
nits = 20_000
nsmp = 10_000

Random.seed!(23462)
settings = Settings(nits, nsmp, 1_000_000, 1.0, 0.0, 1.0, false, true)
priors = BasicPrior(1.0, FixedV(0.1), FixedW(0.5), 0.0, Fixed(), GaussLangevin(1.0,1.0))
@time out1 = pem_sample(state0, dat, priors, settings)

s2 = view(out1["Smp_trans"], 1, :, :)
df2 = DataFrame(hcat(breaks, mean(s2, dims = 2), quantile.(eachrow(s2), 0.025), quantile.(eachrow(s2), 0.25), quantile.(eachrow(s2), 0.75), quantile.(eachrow(s2), 0.975)), :auto)

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
    scale_linetype_manual(values = c("dotdash","solid","dashed","dashed","dotdash")) + ylab("h(t)") + xlab("Time (years)") +
    geom_hline(aes(yintercept = 1 + 1.96), linetype = "dashed", col = cbPalette[6]) +
    geom_hline(aes(yintercept = 1 - 1.96), linetype = "dashed", col = cbPalette[6]) +
    geom_hline(aes(yintercept = 1 + -0.67), linetype = "dotdash", col = cbPalette[4]) +
    geom_hline(aes(yintercept = 1), linetype = "solid", col = cbPalette[7]) +
    geom_hline(aes(yintercept = 1 + 0.67), linetype = "dotdash", col = cbPalette[4])
p2
"""
Random.seed!(12515)
df = CSV.read(datadir("colon.csv"), DataFrame)
y = df.years
n = length(y)
breaks = collect(0.1:0.1:3.1)
p = 1
cens = df.status
covar = fill(1.0, 1, n)
covar = [covar; transpose(df.sex)]
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0), ones(size(x0)))
nits = 20_000
nsmp = 10_000

Random.seed!(23462)
settings = Settings(nits, nsmp, 1_000_000, 1.0, 0.0, 1.0, false, true)
priors = BasicPrior(1.0, FixedV(0.1), FixedW(0.5), 0.0, Fixed(), GaussLangevin(1.0,1.0))
@time out1 = pem_sample(state0, dat, priors, settings)

Random.seed!(23462)
settings = Settings(nits, nsmp, 1_000_000, 1.0, 0.0, 1.0, false, true)
priors = BasicPrior(1.0, FixedV(0.1), FixedW(0.5), 0.0, Fixed(), RandomWalk())
@time out2 = pem_sample(state0, dat, priors, settings)

s1 = view(exp.(out1["Smp_trans"]), 1, :, :)
s2 = view(exp.(out2["Smp_trans"]), 1, :, :)
df2 = DataFrame(hcat(breaks, mean(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975),
            mean(s2, dims = 2), quantile.(eachrow(s2), 0.025), quantile.(eachrow(s2), 0.25), quantile.(eachrow(s2), 0.75), quantile.(eachrow(s2), 0.975)), :auto)

R"""
dat2 = data.frame($df2)
colnames(dat2) <- c("Time","MeanGL","LCIGL","Q1GL","Q4GL","UCIGL","MeanRW","LCIRW","Q1RW","Q4RW","UCIRW") 
"""

R"""
p2 <- dat2 %>%
    pivot_longer(MeanGL:UCIRW) %>%
    ggplot(aes(x = Time, y = value, col = name, linetype = name)) + geom_step() +
    theme_classic() +
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[rep(c(6,7,4,4,6),each=2)]) +
    scale_linetype_manual(values = c(rep(c("solid","dashed"), 5))) + ylab("h(t)") + xlab("Time (years)")
p2
"""


Random.seed!(12515)
df = CSV.read(datadir("colon.csv"), DataFrame)
y = df.years
n = length(y)
breaks = collect(0.0:0.1:3.1)
p = 1
cens = df.status
covar = fill(1.0, 1, n)
#covar = [covar; transpose(df.sex)]
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0), ones(size(x0)))
nits = 50_000
nsmp = 50_000

Random.seed!(23462)
settings = Settings(nits, nsmp, 1_000_000, 1.0, 0.0, 1.0, false, true)
priors = BasicPrior(1.0, FixedV(0.1), FixedW(0.5), 1.0, Fixed(), GaussLangevin(0.0,0.5))
@time out1 = pem_sample(state0, dat, priors, settings)

Random.seed!(23462)
settings = Settings(nits, nsmp, 1_000_000, 1.0, 0.0, 1.0, false, true)
priors = BasicPrior(1.0, FixedV(0.1), FixedW(0.5), 1.0, Fixed(), RandomWalk())
@time out2 = pem_sample(state0, dat, priors, settings)

s1 = view(exp.(out1["Smp_trans"]), 1, :, :)
s2 = view(exp.(out2["Smp_trans"]), 1, :, :)
df2 = DataFrame(hcat(breaks, mean(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975),
            mean(s2, dims = 2), quantile.(eachrow(s2), 0.025), quantile.(eachrow(s2), 0.25), quantile.(eachrow(s2), 0.75), quantile.(eachrow(s2), 0.975)), :auto)

R"""
dat2 = data.frame($df2)
colnames(dat2) <- c("Time","MeanGL","LCIGL","Q1GL","Q4GL","UCIGL","MeanRW","LCIRW","Q1RW","Q4RW","UCIRW") 
"""

R"""
p2 <- dat2 %>%
    pivot_longer(MeanGL:UCIRW) %>%
    ggplot(aes(x = Time, y = value, col = name, linetype = name)) + geom_step() +
    theme_classic() +
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[rep(c(6,7,4,4,6),each=2)]) +
    scale_linetype_manual(values = c(rep(c("solid","dashed"), 5))) + ylab("h(t)") + xlab("Time (years)")
p2
"""

Random.seed!(12515)
df = CSV.read(datadir("colon.csv"), DataFrame)
y = df.years
n = length(y)
breaks = collect(0.0:0.1:3.1)
p = 1
cens = df.status
covar = fill(1.0, 1, n)
#covar = [covar; transpose(df.sex)]
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0), ones(size(x0)))
nits = 100_000
nsmp = 20_000

Random.seed!(3463)
settings = Settings(nits, nsmp, 1_000_000, 2.0, 1.0, 0.5, false, true)
priors = BasicPrior(1.0, FixedV(0.8), FixedW(0.2), 1.0, Cts(20.0, 100.0, 3.4), GaussLangevin(0.0,0.5))
@time out1 = pem_sample(state0, dat, priors, settings)

Random.seed!(346316)
settings = Settings(nits, nsmp, 1_000_000, 1.0, 1.0, 1.0, false, true)
priors = BasicPrior(1.0, FixedV(0.8), FixedW(0.2), 1.0, Fixed(), GaussLangevin(0.0,0.5))
@time out2 = pem_sample(state0, dat, priors, settings)


grd = collect(0.005:0.005:3.1)
t1 = cts_transform(out1["Smp_trans"], out1["Smp_s_loc"], grd)
t2 = cts_transform(out2["Smp_trans"], out2["Smp_s_loc"], grd)
s1 = view(exp.(t1), 1, :, :)
s2 = view(exp.(t2), 1, :, :)
df2 = DataFrame(hcat(grd, mean(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975),
            mean(s2, dims = 2), quantile.(eachrow(s2), 0.025), quantile.(eachrow(s2), 0.25), quantile.(eachrow(s2), 0.75), quantile.(eachrow(s2), 0.975)), :auto)

R"""
dat2 = data.frame($df2)
colnames(dat2) <- c("Time","MeanGL","LCIGL","Q1GL","Q4GL","UCIGL","MeanRW","LCIRW","Q1RW","Q4RW","UCIRW") 
"""

R"""
p2 <- dat2 %>%
    pivot_longer(MeanGL:UCIRW) %>%
    ggplot(aes(x = Time, y = value, col = name, linetype = name)) + geom_step() +
    theme_classic() +
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[rep(c(6,7,4,4,6),each=2)]) +
    scale_linetype_manual(values = c(rep(c("solid","dashed"), 5))) + ylab("h(t)") + xlab("Time (years)")
p2
"""