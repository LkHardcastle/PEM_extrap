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
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 5_000
nsmp = 5_000
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 0.0, 0.1, false, true, 0.05, 10.0)
test_times = [10,50,90.0]
var0 = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5]

Random.seed!(2482)
Barker1 = Vector{Vector{Float64}}()
for i in eachindex(var0)
    priors = BasicPrior(1.0, FixedV([i]), FixedW([0.5]), 1.0, CtsPois(10.0, 10.0, 100.0, 3.2), [GaussLangevin(2.0,2.0)], [], 1.0)
    out1 = pem_fit(state0, dat, priors, settings, test_times)
    push!(Barker1, (sum(out1[1]["Sk_s"][1,:,:], dims = 2)/length(out1[1]["Sk_s"][1,5,:]))[2:end])
end

EM1 = Vector{Vector{Float64}}()
for i in eachindex(var0)
    priors = EulerMaruyama(1.0, FixedV([i]), FixedW([0.5]), 1.0, CtsPois(10.0, 10.0, 100.0, 3.2), [GaussLangevin(2.0,2.0)], [], 1.0)
    out1 = pem_fit(state0, dat, priors, settings, test_times)
    push!(EM1, (sum(out1[1]["Sk_s"][1,:,:], dims = 2)/length(out1[1]["Sk_s"][1,5,:]))[2:end])
end

Barker2 = Vector{Vector{Float64}}()
for i in eachindex(var0)
    priors = BasicPrior(1.0, FixedV([i]), FixedW([0.5]), 1.0, CtsPois(10.0, 10.0, 100.0, 3.2), [GaussLangevin(2.0,0.2)], [], 1.0)
    out1 = pem_fit(state0, dat, priors, settings, test_times)
    push!(Barker2, (sum(out1[1]["Sk_s"][1,:,:], dims = 2)/length(out1[1]["Sk_s"][1,5,:]))[2:end])
end
EM2 = Vector{Vector{Float64}}()
for i in eachindex(var0)
    priors = EulerMaruyama(1.0, FixedV([i]), FixedW([0.5]), 1.0, CtsPois(10.0, 10.0, 100.0, 3.2), [GaussLangevin(2.0,0.2)], [], 1.0)
    out1 = pem_fit(state0, dat, priors, settings, test_times)
    push!(EM2, (sum(out1[1]["Sk_s"][1,:,:], dims = 2)/length(out1[1]["Sk_s"][1,5,:]))[2:end])
end

dfb1 = DataFrame(Barker1, :auto)
rename!(dfb1, Symbol.(var0))
dfb1.Method .= "Barker"

dfem1 = DataFrame(EM1, :auto)
rename!(dfem1, Symbol.(var0))
dfem1.Method .= "Euler-Maruyama"

df1 = vcat(dfb1,dfem1)

CSV.write(datadir("ParamExp1.csv"), df1)

dfb2 = DataFrame(Barker2, :auto)
rename!(dfb2, Symbol.(var0))
dfb2.Method .= "Barker"

dfem2 = DataFrame(EM2, :auto)
rename!(dfem2, Symbol.(var0))
dfem2.Method .= "Euler-Maruyama"
df2 = vcat(dfb2,dfem2)

CSV.write(datadir("ParamExp2.csv"), df2)

R"""
$df1 %>%
    pivot_longer("0.001":"0.5", names_to  = "step_size") %>%
    ggplot(aes(x = step_size, y = value, col = Method)) + geom_boxplot() +
    theme_classic() + scale_colour_manual(values = cbPalette[6:7]) + geom_hline(yintercept = 0.5, linetype = "dotted")
"""

R"""
$df2 %>%
    pivot_longer("0.001":"0.5", names_to  = "step_size") %>%
    ggplot(aes(x = step_size, y = value, col = Method)) + geom_boxplot() +
    theme_classic() + scale_colour_manual(values = cbPalette[6:7]) + geom_hline(yintercept = 0.5, linetype = "dotted")
"""