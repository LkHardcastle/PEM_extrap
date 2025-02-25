using DrWatson
@quickactivate "PEM_extrap"
# For src
using DataStructures, LinearAlgebra, Distributions, Random, Optim, Roots, SpecialFunctions, Statistics
using Plots, CSV, DataFrames, RCall, Interpolations, MCMCDiagnosticTools

include(srcdir("Sampler.jl"))
include(srcdir("PreProcessing.jl"))
include(srcdir("PostProcessing.jl"))

R"""
library(ggplot2)
library(dplyr)
library(tidyr)
library(cowplot)
library(coda)
cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
"""

Random.seed!(2352)
df = CSV.read(datadir("colon.csv"), DataFrame)
y = df.years
maximum(y)
n = length(y)
breaks = collect(0.26:0.25:3.01)
p = 1
cens = df.status
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 1000
nsmp = 10
priors = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPois(1.0, 1.0, 100.0, 3.1), [RandomWalk()], [0.001])
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 100.0)
out1 = pem_fit(state0, dat, priors, settings, test_times)
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 50.0)
out2 = pem_fit(state0, dat, priors, settings, test_times)
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.05, 10.0)
out3 = pem_fit(state0, dat, priors, settings, test_times)
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.1, 5.0)
out4 = pem_fit(state0, dat, priors, settings, test_times)

plot(out1[1]["Sk_σ"][1,:])
plot(out1[1]["Sk_σ"][1,:], out1[1]["Sk_x"][1,1,:])

plot(out1[2]["Sk_v"][1,10,:])
plot(out2[1]["Sk_x"][1,1,:])