using DrWatson
@quickactivate "PEM_extrap"
# For src
using DataStructures, LinearAlgebra, Distributions, Random, Roots, SpecialFunctions
using Plots

include(srcdir("Sampler.jl"))
include(srcdir("PreProcessing.jl"))
include(srcdir("PostProcessing.jl"))

Random.seed!(2436263)
n = 0
y = rand(Exponential(1.0),n)
breaks = collect(0.5:0.5:2.5)
p = 1
cens = fill(1.0,n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = BPS(x0, v0, s0, t0, findall(s0))
priors = ARPrior(1.0, 0.0, FixedW(0.7), 1.0)
#priors = BasicPrior(1.0, FixedV(1.0), FixedW(0.5), 1.0)
nits = 50_000
nsmp = 100000
settings = Settings(nits, nsmp, 100000, 0.5,0.0, 1.0, false, true)
Random.seed!(123)
@time out1 = pem_sample(state0, dat, priors, settings)
settings = Settings(nits, nsmp, 100000, 0.5,0.0, 1.0, false, true)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = BPS(x0, v0, s0, t0, findall(s0))
@time out11 = pem_sample(state0, dat, priors, settings)
Random.seed!(2436263)
settings = Settings(100, nsmp, 100000, 0.5,0.0, 0.1, true, true)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, t0, true, findall(s0))
@time out2 = pem_sample(state0, dat, priors, settings)
@time out21 = pem_sample(state0, dat, priors, settings)

mean(out1["Smp_x"][1,:,:], dims = 2)
mean(out11["Smp_x"][1,:,:], dims = 2)
mean(out21["Smp_x"][1,:,:], dims = 2)
mean(out2["Smp_x"][1,:,:], dims = 2)

quantile.(eachrow(out11["Smp_trans"][1,:,:]), 0.025)
quantile.(eachrow(out1["Smp_trans"][1,:,:]), 0.025)
quantile.(eachrow(out1["Smp_trans"][1,:,:]), 0.975)
quantile.(eachrow(out11["Smp_trans"][1,:,:]), 0.975)
quantile.(eachrow(out21["Smp_trans"][1,:,:]), 0.025)
quantile.(eachrow(out2["Smp_trans"][1,:,:]), 0.025)
quantile.(eachrow(out21["Smp_trans"][1,:,:]), 0.975)
quantile.(eachrow(out2["Smp_trans"][1,:,:]), 0.975)

mean(out1["Smp_x"][1,:,:] .== 0.0, dims = 2)
mean(out11["Smp_x"][1,:,:] .== 0.0, dims = 2)
mean(out2["Smp_x"][1,:,:] .== 0.0, dims = 2)
mean(out21["Smp_x"][1,:,:] .== 0.0, dims = 2)