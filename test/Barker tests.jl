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
breaks = collect(0.1:0.1:1)
p = 1
cens = fill(1.0,n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, t0, true, findall(s0), ones(size(x0)))
nits = 100
nsmp = 100_000
settings = Settings(nits, nsmp, 1_000_000, 0.5,50.0, 0.5, false, true)
priors = BasicPrior(0.5, FixedV(1), FixedW(0.5), 0.0, OU(0.5))
Random.seed!(12515)
@time out1 = pem_sample(state0, dat, priors, settings)

plot(scatter(out1["Smp_t"][1:50],out1["Smp_x"][1,5,1:50]))
plot!(scatter!(out1["Smp_t"][1:50],out1["Smp_x"][1,5,1:50].*out1["Smp_ξ"][1,5,1:50]))
s2 = view(out1["Smp_trans"], 1, :, :)
df2 = DataFrame(hcat(breaks, mean(s2, dims = 2), quantile.(eachrow(s2), 0.025), quantile.(eachrow(s2), 0.25), quantile.(eachrow(s2), 0.75), quantile.(eachrow(s2), 0.975)), :auto)

out1["Smp_x"].*out1["Smp_ξ"]

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
"""