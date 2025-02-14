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

##### Selection procedure using DIC 
DIC = []
Random.seed!(3453)
df = CSV.read(datadir("colon.csv"), DataFrame)
y = df.years
maximum(y)
n = length(y)
breaks = collect(0.25:0.25:3.25)
p = 1
cens = df.status
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 20_000
nsmp = 10_000
settings1 = Exact(nits, nsmp, 1_000_000, 1.0, 10.0, 0.5, false, true)
nits = 300_000
nsmp = 1_000
settings2 = Splitting(nits, nsmp, 1_000_000, 1.0, 0.0, 0.1, false, true, 0.005)

priors1 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 0.0, CtsPois(5.0, 5.0, 150.0, 3.2), [RandomWalk()], [0.0])
priors2 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 0.0, CtsPois(5.0, 5.0, 150.0, 3.2), [RandomWalk()], [0.1])

Random.seed!(24562)
test_times = collect(0.1:0.5:3.1)
@time out1 = pem_fit(state0, dat, priors1, settings1, test_times)
Random.seed!(24562)
@time out2 = pem_fit(state0, dat, priors2, settings2, test_times)

plot(out2[1]["Sk_θ"][1,1,:])
plot(out2[1]["Sk_σ"][1,:])
plot(out1[3])
plot!(out2[3])

plot(out1[4])
plot!(out2[4])

plot!(out1[1]["Sk_x"][1,2,:], log.(out1[1]["Sk_σ"][1,:]))
plot!(out2[1]["Sk_x"][1,2,:], log.(out2[1]["Sk_σ"][1,:]))
plot!(out2[2]["Sk_x"][1,2,:], log.(out2[2]["Sk_σ"][1,:]))

plot(out1[1]["Sk_x"][1,2,:],out1[1]["Sk_x"][1,3,:])
plot!(out2[1]["Sk_x"][1,2,:],out2[1]["Sk_x"][1,3,:])

plot(out1[1]["Sk_θ"][1,2,:],out1[1]["Sk_θ"][1,3,:])
plot!(out2[1]["Sk_θ"][1,2,:],out2[1]["Sk_θ"][1,3,:])

plot(out1[1]["Sk_t"],out1[1]["Sk_θ"][1,2,:])
plot!(out2[1]["Sk_t"],out2[1]["Sk_θ"][1,2,:])

plot(out1[1]["Sk_t"], log.(out1[1]["Sk_σ"][1,:]))
plot!(out2[1]["Sk_t"], log.(out2[1]["Sk_σ"][1,:]))
plot!(out2[2]["Sk_t"], log.(out2[2]["Sk_σ"][1,:]))

histogram(out1[1]["Smp_x"][1,2,:], alpha = 0.1, normalize = true)
histogram!(out2[1]["Sk_x"][1,2,1:200:end], alpha = 0.1, normalize = true)

histogram(out1[1]["Smp_σ"][1,:], alpha = 0.1, normalize = true)
histogram!(out2[1]["Sk_σ"][1,1:200:end], alpha = 0.1, normalize = true)