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
nsmp = 20_000

Random.seed!(23462)
settings = Settings(nits, nsmp, 1_000_000, 2.0, 2.0, 1.0, false, true)
priors1 = BasicPrior(1.0, FixedV([0.2]), FixedW([0.8]), 1.0, CtsPois(10.0, 150.0, 3.2), [RandomWalk()])
@time out1 = pem_sample(state0, dat, priors1, settings)

plot(out1["Sk_t"],out1["Sk_x"][1,10,:])
mean(out1["Smp_x"][1,1,:])
mean(out1["Smp_x"][1,10,:] .== 0.0)
quantile(out1["Smp_x"][1,5,:], 0.975)
quantile(out1["Smp_x"][1,10,:], 0.025)
quantile(Normal(0,1), 0.025)



plot(scatter(log.(out1["Sk_σ"][1,:]), out1["Sk_x"][1,3,:].*out1["Sk_σ"][1,:]))
plot(scatter(log.(out1["Sk_σ"][1,:]), out1["Sk_x"][1,3,:]))
plot(log.(out1["Sk_σ"][1,10_000:end]), out1["Sk_x"][1,3,10_000:end])
plot(out1["Sk_t"],out1["Sk_σ"][1,:])
out1["Smp_J"]
plot(out1["Sk_t"],out1["Sk_x"][1,1,:])
plot!(out1["Sk_t"],out1["Sk_v"][1,1,:])