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

## TODO - For Random Walk run Poisson vs Negative Binomial.

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
nits = 20_000
nsmp = 10

settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 50.0)

test_Gamma = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0]
DIC = []
mean_obs = []
Random.seed!(23521)
Gamma_used = []
test_times = [0.5, 1.5, 2.5]
for Γ_ in test_Gamma
    x0, v0, s0 = init_params(p, dat)
    v0 = v0./norm(v0)
    priors = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPois(Γ_, 1.0, 100.0, 3.1), [RandomWalk()], [0.1])
    out = pem_fit(state0, dat, priors, settings, test_times)
    println(out[3]);println(out[4])
    # Get DIC 
    push!(DIC,get_DIC(out[1], dat, 1_000)[2])
    push!(DIC,get_DIC(out[2], dat, 1_000)[2])
    push!(Gamma_used, Γ_)
    push!(Gamma_used, Γ_)
end

out = copy(out[1])
test_a = [1.0, 2.5, 5.0, 10.0]
test_b = [0.1, 0.5, 1.0, 2.5]
DIC = []
mean_obs = []
a_used = []
b_used = []
Random.seed!(23524)
for a_ in test_a
    for b_ in test_b
        x0, v0, s0 = init_params(p, dat)
        v0 = v0./norm(v0)
        priors = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB(a_, b_, a_/b_, 150.0, 3.02), [RandomWalk()], [0.1])
        out = pem_fit(state0, dat, priors, settings, test_times)
        push!(a_used, a_)
        push!(b_used, b_)
        # Get DIC
        
        # Get Mean survival

    end
end

## TODO - For optimal models re-run for Gauss/Gamma/Gompertz
Random.seed!(2352)
df = CSV.read(datadir("colon.csv"), DataFrame)
y = df.years
maximum(y)
n = length(y)
breaks = collect(0.25:0.25:3.0)
p = 1
cens = df.status
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 100_000
nsmp = 10_000
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.05, 30.0)


## TODO - Report mean survival for optimal Poisson/NB model for each set of priors



## TODO - Run comparators and compare mean survival/hazards

