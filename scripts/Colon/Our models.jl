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

Random.seed!(9102)
df = CSV.read(datadir("colon.csv"), DataFrame)
y = df.years
maximum(y)
n = length(y)
breaks = collect(0.1:0.1:3.1)
p = 1
cens = df.status
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 150000
nsmp = 20000
settings = Settings(nits, nsmp, 1_000_000, 1.0,0.5, 0.5, false, true)


priors1 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [10.0], [10.0]), 1.0, Cts(10.0, 50.0, 3.5), [RandomWalk()])
priors2 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [10.0], [10.0]), 1.0, Cts(10.0, 50.0, 3.5), [GaussLangevin(-1.0,1.0)])
priors3 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [10.0], [10.0]), 1.0, Cts(10.0, 50.0, 3.5), [GammaLangevin(0.5,2)])
priors4 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [10.0], [10.0]), 1.0, Cts(10.0, 50.0, 3.5), [GompertzBaseline(0.5)])
Random.seed!(9102)
@time out1 = pem_sample(state0, dat, priors1, settings)
@time out2 = pem_sample(state0, dat, priors2, settings)
@time out3 = pem_sample(state0, dat, priors3, settings)
@time out4 = pem_sample(state0, dat, priors4, settings)