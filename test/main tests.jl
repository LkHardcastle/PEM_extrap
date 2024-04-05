using DrWatson
@quickactivate "PEM_extrap"
# For src
using DataStructures, LinearAlgebra, Distributions, Random
using Plots

include(srcdir("PreProcessing.jl"))
include(srcdir("Sampler.jl"))
include(srcdir("PostProcessing.jl"))

Random.seed!(123)
n = 100
y = rand(Exponential(0.5),n)
maximum(y)
breaks = collect(0.05:0.05:(maximum(y) + 0.1))
p = 1
x0, v0, s0 = init_params(breaks, p)
cens = rand(Bernoulli(0.9),n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
t0 = 0.0
priors = FixedPrior(fill(0.5, size(x0)), 1.0, 1.0, 0.0)
nits = 1_000_000
settings = Settings(nits, 0.5, false)
Random.seed!(123)
out = @time pem_sample(x0, s0, v0, t0, dat, priors, settings)

plot(out["t"][1:nits], vec(out["Sk_x"][:,1,:])[1:nits])
plot!(out["t"][1:nits], vec(out["Sk_x"][:,2,:])[1:nits])
plot!(out["t"][1:100], vec(out["Sk_x"][:,3,:])[1:100])
plot!(out["t"][1:100], vec(out["Sk_x"][:,4,:])[1:100])
plot!(out["t"][1:100], vec(out["Sk_x"][:,5,:])[1:100])
plot!(out["t"][1:100], vec(out["Sk_x"][:,6,:])[1:100])
plot(out["t"][1:100], vec(out["Sk_x"][:,7,:])[1:100])
plot!(out["t"][1:100], vec(out["Sk_x"][:,8,:])[1:100])

plot(out["t"],vec(sum(out["Sk_s"], dims = 2)))

smps = post_estimates(out, dat, collect(1:3:out["t"][end]))

smps
plot(out["t"][1:nits], vec(out["Sk_x"][:,2,:])[1:nits])
plot(out["t"][1:nits], vec(out["Sk_x"][:,56,:])[1:nits])

plot(eachcol(smps), layout = 8)
plot(breaks, vec(exp.(median(smps, dims = 1))))
plot!(breaks, vec(exp.(quantile.(eachcol(smps), 0.05))))
plot!(breaks, vec(exp.(quantile.(eachcol(smps), 0.95))))
out["Eval"]