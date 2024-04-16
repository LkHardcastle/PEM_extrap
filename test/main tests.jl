using DrWatson
@quickactivate "PEM_extrap"
# For src
using DataStructures, LinearAlgebra, Distributions, Random
using Plots

include(srcdir("Sampler.jl"))
include(srcdir("PreProcessing.jl"))
include(srcdir("PostProcessing.jl"))

Random.seed!(123)
n = 0
y = rand(Exponential(0.5),n)
#maximum(y)
#breaks = collect(0.05:0.05:(maximum(y) + 0.1))
#breaks = collect(0.05:0.05:1.0)
breaks = [0.5, 1.0]
p = 1
cens = fill(1.0,n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
t0 = 0.0
priors = FixedPrior(fill(0.5, size(x0)), 1.0, 1.0, 0.0)
nits = 1_000_000
settings = Settings(nits, 0.9, 0.0, false)
Random.seed!(4583)
out1 = @time pem_sample(x0, s0, v0, t0, dat, priors, settings)
x0, v0, s0 = init_params(p, dat)
out2 = @time pem_sample(x0, s0, v0, t0, dat, priors, settings)
x0, v0, s0 = init_params(p, dat)
out3 = @time pem_sample(x0, s0, v0, t0, dat, priors, settings)

smps1 = post_estimates(out1, dat, collect(100000.0:10.0:out1["t"][end]))
smps2 = post_estimates(out2, dat, collect(100000.0:10.0:out2["t"][end]))
smps3 = post_estimates(out3, dat, collect(100000.0:10.0:out3["t"][end]))

mean(smps1[:,1])
mean(smps2[:,1])
mean(smps3[:,1])
quantile(smps1[:,1],0.025)
quantile(smps2[:,1],0.025)
quantile(smps3[:,1],0.025)
quantile(Normal(0,1),0.025)
quantile(smps1[:,1],0.975)
quantile(smps2[:,1],0.975)
quantile(smps3[:,1],0.975)
quantile(Normal(0,1),0.975)
mean(smps1[:,2])
mean(smps2[:,2])
mean(smps3[:,2])


mean(smps1[findall(smps1[:,1] .== smps1[:,2]),1])
mean(smps2[findall(smps2[:,1] .== smps2[:,2]),1])
mean(smps3[findall(smps3[:,1] .== smps3[:,2]),1])
quantile(smps1[findall(smps1[:,1] .== smps1[:,2]),1],0.025)
quantile(smps2[findall(smps2[:,1] .== smps2[:,2]),1],0.025)
quantile(smps3[findall(smps3[:,1] .== smps3[:,2]),1],0.025)
quantile(smps1[findall(smps1[:,1] .== smps1[:,2]),1],0.975)
quantile(smps2[findall(smps2[:,1] .== smps2[:,2]),1],0.975)
quantile(smps3[findall(smps3[:,1] .== smps3[:,2]),1],0.975)


mean(smps1[findall(smps1[:,1] .!= smps1[:,2]),1])
mean(smps2[findall(smps2[:,1] .!= smps2[:,2]),1])
mean(smps3[findall(smps3[:,1] .!= smps3[:,2]),1])
quantile(smps1[findall(smps1[:,1] .!= smps1[:,2]),1],0.025)
quantile(smps2[findall(smps2[:,1] .!= smps2[:,2]),1],0.025)
quantile(smps3[findall(smps3[:,1] .!= smps3[:,2]),1],0.025)
quantile(smps1[findall(smps1[:,1] .!= smps1[:,2]),1],0.975)
quantile(smps2[findall(smps2[:,1] .!= smps2[:,2]),1],0.975)
quantile(smps3[findall(smps3[:,1] .!= smps3[:,2]),1],0.975)

quantile(Normal(0,sqrt(2)),0.025)
mean(smps1[findall(smps1[:,1] .!= smps1[:,2]),2])
mean(smps2[findall(smps2[:,1] .!= smps2[:,2]),2])
mean(smps3[findall(smps3[:,1] .!= smps3[:,2]),2])
quantile(smps1[findall(smps1[:,1] .!= smps1[:,2]),2],0.025)
quantile(smps2[findall(smps2[:,1] .!= smps2[:,2]),2],0.025)
quantile(smps3[findall(smps3[:,1] .!= smps3[:,2]),2],0.025)
quantile(smps1[findall(smps1[:,1] .!= smps1[:,2]),2],0.975)
quantile(smps2[findall(smps2[:,1] .!= smps2[:,2]),2],0.975)
quantile(smps3[findall(smps3[:,1] .!= smps3[:,2]),2],0.975)


mean(smps1[:,1] .== smps1[:,2])
mean(smps2[:,1] .== smps2[:,2])
mean(smps3[:,1] .== smps3[:,2])

mean(smps[findall(smps[:,1] .!= smps[:,2]),1] - smps[findall(smps[:,1] .!= smps[:,2]),2])
quantile(smps[findall(smps[:,1] .!= smps[:,2]),1] - smps[findall(smps[:,1] .!= smps[:,2]),2], 0.975)
quantile(smps[findall(smps[:,1] .!= smps[:,2]),1] - smps[findall(smps[:,1] .!= smps[:,2]),2], 0.025)
quantile(Normal(0,sqrt(2)),0.975)

out1["Eval"]

plot(out1["t"][1:nits], vec(out1["Sk_x"][:,1,:])[1:nits])
plot!(out1["t"][1:nits], vec(out1["Sk_x"][:,2,:])[1:nits])
plot!(out1["t"][1:20], vec(out1["Sk_v"][:,1,:])[1:20])
plot!(out1["t"][1:20], vec(out1["Sk_v"][:,2,:])[1:20])
