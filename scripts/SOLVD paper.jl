using DrWatson
@quickactivate "PEM_extrap"
# For src
using DataStructures, LinearAlgebra, Distributions, Random, Optim, Roots, SpecialFunctions
using Plots, CSV, DataFrames, RCall, Interpolations, MCMCDiagnosticTools

include(srcdir("Sampler.jl"))
include(srcdir("PreProcessing.jl"))
include(srcdir("PostProcessing.jl"))

R"""
library(ggplot2)
library(dplyr)
library(tidyr)
library(cowplot)
library(ggsurvfit)
library(survival)
cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
"""

Random.seed!(9102)
df_all = CSV.read(datadir("SOLVD","SOLVD.csv"), DataFrame)
df = filter(idx -> idx.TRIAL == "T" && idx.EPYTIME > 0.0, df_all)
sum(df.EPX)
y = df.EPYTIME/365
maximum(y)
minimum(y)
n = length(y)
breaks = collect(0.02:0.1:4.62)
p = 1
cens = df.EPX
covar = fill(1.0, 1, n)
trt = (df.DRUG .== "P").*1.0
covar = [covar; transpose(trt)]
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
x0[2,:] = vcat(x0[2,1], zeros(size(breaks) .-1))
v0[2,:] = vcat(v0[2,1], 1.0, zeros(size(breaks) .-2))
s0[2,:] = vcat(s0[2,1], true, zeros(Int,size(breaks) .-2))
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 1_000
nsmp = 10
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 50.0)
test_times = collect(0.2:0.2:3.0)

priors = BasicPrior(1.0, PC([1.0, 1.0], [2, 2], [0.5, 0.5], Inf), FixedW([0.5, 0.5]), 1.0, CtsPois(14.0, 1.0, 200.0, 4.65), [RandomWalk(), RandomWalk()], [0.01, 0.01], 2)
out1 = pem_fit(state0, dat, priors, settings, test_times)
println(out1[3]);println(out1[4])