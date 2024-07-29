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
breaks = collect(0.5:0.5:5)
p = 1
cens = fill(1.0,n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0

Random.seed!(34734)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
state0 = ECMC2(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks), true, findall(s0))
nits = 200000
nsmp = 100000
settings = Settings(nits, nsmp, 1_000_000, 0.5,0.5, 0.5, false, true)
priors = BasicPrior(1.0, FixedV(1.0), FixedW(0.5), 1.0, Cts(3.0, 50.0, 6.0))
@time out1 = pem_sample(state0, dat, priors, settings)

test = rand(Poisson(6*3), 10_000) .+ 1
histogram(test, norm = true, bins = 100)
histogram!(out1["Smp_J"].+ 0.5, norm = true, bins = 100) 


grid = collect(0.05:0.05:5)
test_smp = cts_transform(out1["Smp_trans"], out1["Smp_s_loc"], grid)

s2 = view(test_smp, 1, :, :)
df2 = DataFrame(hcat(grid, mean(s2, dims = 2), quantile.(eachrow(s2), 0.025), quantile.(eachrow(s2), 0.25), quantile.(eachrow(s2), 0.75), quantile.(eachrow(s2), 0.975)), :auto)
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
    scale_linetype_manual(values = c("dotdash","solid","dashed","dashed","dotdash")) + ylab("h(t)") + xlab("Time (years)")
p2
#plot_grid(p1,p2)
#ggsave($plotsdir("SnS.pdf"), width = 14, height = 6)
"""



Random.seed!(123)
df = CSV.read(datadir("colon.csv"), DataFrame)
y = df.years
maximum(y)
n = length(y)
breaks = collect(0.1:0.1:3.1)
p = 1
cens = df.status
covar = fill(1.0, 1, n)
covar = [covar; transpose(df.sex)]
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
x0[2,:] = vcat(x0[2,1],zeros(size(breaks) .-1))
v0[2,:] = vcat(v0[2,1],zeros(size(breaks) .-1))
s0[2,:] = vcat(s0[2,1],zeros(Int,size(breaks) .-1))
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks), true, findall(s0))
nits = 100000
nsmp = 100000
settings = Settings(nits, nsmp, 1_000_000, 0.5,0.5, 0.5, false, true)
#priors = BasicPrior(1.0, FixedV(1.0), FixedW(0.5), 1.0, Cts(3.0, 50.0, 6.0))
priors = BasicPrior(1.0, FixedV(1.0), FixedW(0.5), 1.0, Cts(3.0, 50.0, 3.5))
@time out1 = pem_sample(state0, dat, priors, settings)

grid = collect(0.01:0.01:3.2)
test_smp = cts_transform(out1["Smp_trans"], out1["Smp_s_loc"], grid)
s2 = view(exp.(test_smp), 1, :, :)
s3 = exp.(test_smp[1,:,:] .+ test_smp[2,:,:])
df2 = DataFrame(hcat(grid, mean(s2, dims = 2), mean(s3, dims = 2), quantile.(eachrow(s2), 0.025), quantile.(eachrow(s2), 0.25), quantile.(eachrow(s2), 0.75), quantile.(eachrow(s2), 0.975)), :auto)
R"""
dat2 = data.frame($df2)
colnames(dat2) <- c("Time","Mean","Mean2","LCI","Q1","Q4","UCI") 
"""
R"""    
p2 <- dat2 %>%
    pivot_longer(Mean:Mean2) %>%
    ggplot(aes(x = Time, y = value, col = name, linetype = name)) + geom_step() +
    theme_classic() +
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,4,4,6)]) +
    scale_linetype_manual(values = c("dotdash","solid","dashed","dashed","dotdash")) + ylab("h(t)") + xlab("Time (years)")
p2
#plot_grid(p1,p2)
#ggsave($plotsdir("SnS.pdf"), width = 14, height = 6)
"""