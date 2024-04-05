using DrWatson
@quickactivate "PEM_extrap"
# For src
using DataStructures, LinearAlgebra, Distributions, Random

include(srcdir("PreProcessing.jl"))
include(srcdir("Sampler.jl"))

Random.seed!(123)
n = 100
y = rand(Exponential(0.5),n)
maximum(y)
breaks = collect(0.05:0.05:4)
p = 1
x0, v0, s0 = init_params(breaks, p)
cens = rand(Bernoulli(0.9),n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)