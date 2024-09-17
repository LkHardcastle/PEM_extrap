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
breaks = collect(0.1:0.1:3.0)
p = 1
cens = fill(1.0,n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0))
nits = 50_000
nsmp = 25_000



priors1 = BasicPrior(0.2, FixedV([0.2]), FixedW([0.5]), 1.0, Cts(5.0, 100.0, 3.2), [RandomWalk()])
priors2 = BasicPrior(0.2, FixedV([0.2]), FixedW([0.5]), 0.0, RJ(5.0, 0.1, 100.0, 3.2), [RandomWalk()])
priors3 = BasicPrior(0.2, FixedV([0.2]), FixedW([0.5]), 0.0, RJ(5.0, 0.1, 100.0, 3.2), [RandomWalk()])
priors4 = BasicPrior(0.2, FixedV([0.2]), FixedW([0.5]), 0.0, RJ(5.0, 2.0, 100.0, 3.2), [RandomWalk()])

Random.seed!(9102)
settings = Settings(nits, nsmp, 1_000_000, 5.0, 0.1, 1.0, false, true)
@time out1 = pem_sample(state0, dat, priors1, settings)
Random.seed!(9102)
settings = Settings(nits, nsmp, 1_000_000, 5.0, 10.0, 1.0, false, true)
@time out2 = pem_sample(state0, dat, priors2, settings)
Random.seed!(9102)
@time out3 = pem_sample(state0, dat, priors3, settings)
Random.seed!(9102)
@time out4 = pem_sample(state0, dat, priors4, settings)


histogram(out1["Smp_J"])
histogram(out2["Smp_J"])

mean(out1["Smp_J"])
plot(out1["Smp_J"])
mean(sum(out1["Smp_s"],dims = 2))
plot(sum(out1["Smp_s"],dims = 2)[1,1,:])
mean(out2["Smp_J"])
plot(out2["Smp_J"])

mean(sum(out1["Smp_s"],dims = 2))
mean(out2["Smp_J"])
mean(out3["Smp_J"])
mean(out4["Smp_J"])
mean(out5["Smp_J"])
mean(out6["Smp_J"])



plot(log.(sum(out1["Smp_s"],dims = 2)[1,1,:]))
plot!(log.(out2["Smp_J"]))
plot!(log.(out3["Smp_J"]))
plot!(out4["Smp_J"])
plot!(out5["Smp_J"])

plot(out1["Smp_x"][1,2,:])
plot(out2["Smp_x"][1,2,:])
plot(out3["Smp_x"][1,4,:])

out3["Smp_s_loc"]

sum(out2["Smp_v"][1,:,5_000])
plot(sum(out2["Smp_v"][1,1:out2["Smp_J"][1:end],1:end].^2))

out1["Smp_s_loc"]

histogram(sum(out1["Smp_s"],dims = 2)[1,1, 1_000:end], normalize = :probability)
histogram!(out2["Smp_J"], normalize = :probability)
plot!(collect(2:31),pdf.(Poisson(3.1*2.5),1:30)/(1-pdf(Poisson(3.1*2.5),0)))