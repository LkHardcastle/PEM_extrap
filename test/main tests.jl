using DrWatson
@quickactivate "PEM_extrap"
# For src
using DataStructures, LinearAlgebra, Distributions, Random
using Plots

include(srcdir("PreProcessing.jl"))
include(srcdir("Sampler.jl"))
include(srcdir("PostProcessing.jl"))

Random.seed!(123)
n = 0
y = rand(Exponential(0.5),n)
#maximum(y)
#breaks = collect(0.05:0.05:(maximum(y) + 0.1))
#breaks = collect(0.05:0.05:1.0)
breaks = [0.5, 1.0]
p = 1
x0, v0, s0 = init_params(breaks, p)
cens = rand(Bernoulli(0.9),n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
t0 = 0.0
priors = FixedPrior(fill(0.5, size(x0)), 1.0, 1.0, 0.0)
nits = 2_000_000
settings = Settings(nits, 0.5, false, 0.0)
Random.seed!(123)
out1 = @time pem_sample(x0, s0, v0, t0, dat, priors, settings)

priors = HyperPrior(fill(0.1, size(x0)), 0.1, 2.5, 0.2, 1.0, 0.0)
settings = Settings(nits, 0.5, 2.0, false)
Random.seed!(123)
out2 = @time pem_sample(x0, s0, v0, t0, dat, priors, settings)

priors = HyperPrior2(fill(0.1, size(x0)), 0.1, 2, 5, 0.2, 1.0, 0.0)
settings = Settings(nits, 0.5, 2.0, false)
Random.seed!(123)
out3 = @time pem_sample(x0, s0, v0, t0, dat, priors, settings)

smps = post_estimates(out1, dat, collect(100.0:100.0:out1["t"][end]))

mean(smps[:,1])
quantile(smps[:,1],0.025)
quantile(smps[:,1],0.975)
mean(smps[:,2])
quantile(smps[:,2],0.025)
quantile(smps[:,2],0.975)
mean(smps[findall(smps[:,1] .== smps[:,2]),1])
quantile(smps[findall(smps[:,1] .== smps[:,2]),1],0.025)
quantile(smps[findall(smps[:,1] .== smps[:,2]),1],0.975)
mean(smps[findall(smps[:,1] .!= smps[:,2]),1])
quantile(smps[findall(smps[:,1] .!= smps[:,2]),1],0.025)
quantile(smps[findall(smps[:,1] .!= smps[:,2]),1],0.975)
mean(smps[findall(smps[:,1] .!= smps[:,2]),2])
quantile(smps[findall(smps[:,1] .!= smps[:,2]),2],0.025)
quantile(smps[findall(smps[:,1] .!= smps[:,2]),2],0.975)
plot(scatter(smps[:,1],smps[:,2]))
mean(smps[:,1] .== smps[:,2])

mean(smps[findall(smps[:,1] .!= smps[:,2]),1] - smps[findall(smps[:,1] .!= smps[:,2]),2])
quantile(smps[findall(smps[:,1] .!= smps[:,2]),1] - smps[findall(smps[:,1] .!= smps[:,2]),2], 0.975)
quantile(smps[findall(smps[:,1] .!= smps[:,2]),1] - smps[findall(smps[:,1] .!= smps[:,2]),2], 0.025)
quantile(Normal(0,sqrt(2)),0.975)

histogram(smps[:,1])
mean(smps[:,2])
mean(smps[:,3])

plot(out3["Sk_h"])
plot(out2["Sk_h"])
out2["Eval"]

plot(out1["t"][1:200], vec(out1["Sk_x"][:,1,:])[1:200])
plot!(out1["t"][1:200], vec(out1["Sk_x"][:,2,:])[1:200])

histogram(vec(out1["Sk_x"][:,1,:])[1:nits])
quantile(vec(out1["Sk_x"][:,1,:])[1:nits],0.05)
quantile(vec(out1["Sk_x"][:,1,:])[1:nits],0.95)
plot(out3["t"][1:nits], vec(out3["Sk_x"][:,56,:])[1:nits])

mean(vec(sum(out1["Sk_s"], dims = 2)))
mean(vec(sum(out2["Sk_s"], dims = 2)))
mean(vec(sum(out3["Sk_s"], dims = 2)))
plot(out1["t"],vec(sum(out1["Sk_s"], dims = 2)))
plot(out2["t"],vec(sum(out2["Sk_s"], dims = 2)))
plot(out3["t"],vec(sum(out3["Sk_s"], dims = 2)))


plot!(out["t"][1:nits], vec(out["Sk_x"][:,2,:])[1:nits])
plot!(out["t"][1:100], vec(out["Sk_x"][:,3,:])[1:100])
plot!(out["t"][1:100], vec(out["Sk_x"][:,4,:])[1:100])
plot!(out["t"][1:100], vec(out["Sk_x"][:,5,:])[1:100])
plot!(out["t"][1:100], vec(out["Sk_x"][:,6,:])[1:100])
plot(out["t"][1:100], vec(out["Sk_x"][:,7,:])[1:100])
plot!(out["t"][1:100], vec(out["Sk_x"][:,8,:])[1:100])

plot(out["t"],vec(sum(out["Sk_s"], dims = 2)))
histogram(vec(sum(out["Sk_s"], dims = 2)))

smps = post_estimates(out, dat, collect(1:3:out["t"][end]))

smps
plot(out["t"][1:nits], vec(out["Sk_x"][:,2,:])[1:nits])
plot(out["t"][1:nits], vec(out["Sk_x"][:,56,:])[1:nits])

plot(eachcol(smps), layout = 8)
plot(breaks, vec(exp.(median(smps, dims = 1))))
plot!(breaks, vec(exp.(quantile.(eachcol(smps), 0.05))))
plot!(breaks, vec(exp.(quantile.(eachcol(smps), 0.95))))
out["Eval"]