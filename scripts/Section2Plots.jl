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
breaks = collect(1:1:300)
p = 1
cens = fill(1.0,n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0))
nits = 50_000
nsmp = 20_000

Random.seed!(23462)
settings = Settings(nits, nsmp, 1_000_000, 2.0, 2.0, 1.0, false, true)
priors1 = BasicPrior(1.0, FixedV([0.2]), FixedW([0.5]), 0.0, Fixed(0.1), [GaussLangevin(1.0,1.0)])
@time out1 = pem_sample(state0, dat, priors1, settings)
priors2 = BasicPrior(1.0, FixedV([0.2]), FixedW([0.5]), 0.0, Fixed(0.1), [GammaLangevin(2.0,1.0)])
@time out2 = pem_sample(state0, dat, priors2, settings)


s1 = view(exp.(cumsum(out1["Smp_x"], dims = 2)), 1, :, :)
df1 = DataFrame(hcat(breaks, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

s1 = view(exp.(cumsum(out2["Smp_x"], dims = 2)), 1, :, :)
df2 = DataFrame(hcat(breaks, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

R"""
dat1 = data.frame($df1)
dat1 = cbind(dat1, "Log-Normal stationary")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat2 = data.frame($df2)
dat2 = cbind(dat2, "Gamma stationary")
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat_diffusion <- rbind(dat1, dat2)
"""

R"""
p3 <- dat_diffusion %>%
    pivot_longer(c(Mean, Q1, Q4)) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 1), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7)]) +
    scale_linetype_manual(values = c("solid","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (arbitrary units)") + ylim(0,NA)
ggsave($plotsdir("Priors.pdf"), width = 8, height = 4)
"""

Random.seed!(12515)
n = 0
y = rand(Exponential(1.0),n)
breaks = collect(1:1:300)
p = 1
cens = fill(1.0,n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0))
nits = 50_000
nsmp = 20_000

Random.seed!(23462)
settings = Settings(nits, nsmp, 1_000_000, 2.0, 2.0, 1.0, false, true)
priors1 = BasicPrior(1.0, FixedV([0.2]), FixedW([0.5]), 0.0, Fixed(0.1), [RandomWalk()])
@time out1 = pem_sample(state0, dat, priors1, settings)
priors2 = BasicPrior(1.0, FixedV([0.2]), FixedW([0.5]), 0.0, Fixed(0.1), [GaussLangevin(2.0,1.0)])
@time out2 = pem_sample(state0, dat, priors2, settings)
priors3 = BasicPrior(1.0, FixedV([0.2]), FixedW([0.5]), 0.0, Fixed(0.1), [GompertzBaseline(0.5)])
@time out3 = pem_sample(state0, dat, priors2, settings)

R"""
dat1 = data.frame($df1)
dat1 = cbind(dat1, "Log-Normal stationary")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat2 = data.frame($df2)
dat2 = cbind(dat2, "Gamma stationary")
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat_diffusion <- rbind(dat1, dat2)
"""

R"""
p3 <- dat_diffusion %>%
    pivot_longer(c(Mean, Q1, Q4)) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 1), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7)]) +
    scale_linetype_manual(values = c("solid","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (arbitrary units)") + ylim(0,NA)
ggsave($plotsdir("Priors.pdf"), width = 8, height = 4)
"""