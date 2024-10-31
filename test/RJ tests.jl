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
nits = 100_000
nsmp = 200_000



priors1 = BasicPrior(0.2, FixedV([0.2]), FixedW([0.5]), 1.0, Cts(5.0, 100.0, 3.1), [RandomWalk()])
priors2 = BasicPrior(0.2, FixedV([0.2]), FixedW([0.5]), 0.0, RJ(5.0, 0.2, 100.0, 3.1), [RandomWalk()])
priors3 = BasicPrior(0.2, FixedV([0.2]), FixedW([0.5]), 0.0, RJ(5.0, 0.2, 100.0, 3.1), [RandomWalk()])
priors4 = BasicPrior(0.2, FixedV([0.2]), FixedW([0.5]), 0.0, RJ(5.0, 2.0, 100.0, 3.1), [RandomWalk()])

nits = 100_000
nsmp = 20_000
Random.seed!(9102)
settings = Settings(nits, nsmp, 1_000_000, 5.0, 1.0, 1.0, false, true)
state0 = ECMC2(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0))
@time out1 = pem_sample(state0, dat, priors1, settings)
Random.seed!(9102)
state0 = ECMC2(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0))
settings = Settings(nits, nsmp, 1_000_000, 5.0, 1.0, 1.0, false, true)
@time out2 = pem_sample(state0, dat, priors2, settings)
Random.seed!(9102)
settings = Settings(nits, nsmp, 5_000_000, 0.1, 1.0, 1.0, false, true)
state0 = RWM(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0), 0.05, 0)
@time out3 = pem_sample(state0, dat, priors3, settings)

out2["Smp_x"]
logpdf(Normal(0,))

histogram(out1["Smp_J"])
histogram(out2["Smp_J"])

mean(out1["Smp_J"])
plot(out1["Smp_J"])
mean(sum(out1["Smp_s"],dims = 2))
plot(sum(out1["Smp_s"],dims = 2)[1,1,:])
mean(sum(out1["Smp_s"],dims = 2)[1,1,:])
mean(out2["Smp_J"])
mean(out3["Sk_J"])
plot!(out2["Smp_J"])
plot!(out3["Sk_J"])
plot(out1["Smp_x"][1,1,:])
plot(out3["Sk_x"][1,1,:])



histogram(out3["Sk_x"][1,5,:])
plot!(-0.75:0.001:0.75, 2300 .*pdf.(Normal(0.0,0.2), -0.75:0.001:0.75))


histogram(out1["Smp_x"][1,9,:])
plot!(-0.75:0.001:0.75, 2000 .*pdf.(Normal(0.0,0.2), -0.75:0.001:0.75))

histogram(out3["Sk_J"])
plot(pdf.(Poisson(7.5), 1:20))
length(findall(out3["Sk_J"] .> 10))/length(out3["Sk_J"])
length(findall(out3["Sk_J"] .< 5))/length(out3["Sk_J"])
length(findall(out3["Sk_J"] .== 5))/length(out3["Sk_J"])
length(findall(out3["Sk_J"] .== 7))/length(out3["Sk_J"])
length(findall(sum(out1["Smp_s"],dims = 2)[1,1,:] .> 10))/length(sum(out1["Smp_s"],dims = 2)[1,1,:])
length(findall(sum(out1["Smp_s"],dims = 2)[1,1,:] .< 5))/length(sum(out1["Smp_s"],dims = 2)[1,1,:])
length(findall(sum(out1["Smp_s"],dims = 2)[1,1,:] .== 5))/length(sum(out1["Smp_s"],dims = 2)[1,1,:])
length(findall(sum(out1["Smp_s"],dims = 2)[1,1,:] .== 7))/length(sum(out1["Smp_s"],dims = 2)[1,1,:])

length(findall(sum(out2["Smp_s"],dims = 2)[1,1,:] .> 10))/length(sum(out2["Smp_s"],dims = 2)[1,1,:])
length(findall(sum(out2["Smp_s"],dims = 2)[1,1,:] .< 5))/length(sum(out2["Smp_s"],dims = 2)[1,1,:])
length(findall(sum(out2["Smp_s"],dims = 2)[1,1,:] .== 5))/length(sum(out2["Smp_s"],dims = 2)[1,1,:])
length(findall(sum(out2["Smp_s"],dims = 2)[1,1,:] .== 7))/length(sum(out2["Smp_s"],dims = 2)[1,1,:])
1 - cdf(Poisson(7.5), 9)
cdf(Poisson(7.5), 4)
pdf(Poisson(7.5),4)
pdf(Poisson(7.5),6)
