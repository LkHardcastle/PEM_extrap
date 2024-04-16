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
priors = FixedPrior(fill(0.5, size(x0)), 2.0, 2.0, 0.0)
nits = 400_000
nsmp = 10_000
settings = Settings(nits, nsmp, 0.9, 0.5, 0.0, false)
Random.seed!(6346)
out1 = @time pem_sample(x0, s0, v0, t0, dat, priors, settings)
nits = 400_000
nsmp = 20_000
settings = Settings(nits, nsmp, 0.9, 0.5, 0.0, false)
Random.seed!(26262626)
x0, v0, s0 = init_params(p, dat)
out2 = @time pem_sample(x0, s0, v0, t0, dat, priors, settings)
nits = 400_000
nsmp = 40_000
settings = Settings(nits, nsmp, 0.9, 0.5, 0.0, false)
Random.seed!(4682)
x0, v0, s0 = init_params(p, dat)
out3 = @time pem_sample(x0, s0, v0, t0, dat, priors, settings)

smps1 = out1["Smp_x"]
smps2 = out2["Smp_x"]
smps3 = out3["Smp_x"]

v1 = out1["Smp_v"]
histogram(v1[:,1])
histogram(v1[:,2])

histogram(vec(out1["Sk_x"][:,1,1:10_000]))
histogram(vec(out1["Sk_x"][:,2,1:10_000]))

mean(smps1[:,1])
mean(smps2[:,1])
mean(smps3[:,1])
quantile(smps1[:,1],0.025)
quantile(smps2[:,1],0.025)
quantile(smps3[:,1],0.025)
quantile(Normal(0,2),0.025)
quantile(smps1[:,1],0.975)
quantile(smps2[:,1],0.975)
quantile(smps3[:,1],0.975)
quantile(Normal(0,2),0.975)
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


mean(smps1[findall(smps1[:,1] .!= smps1[:,2]),2])
mean(smps2[findall(smps2[:,1] .!= smps2[:,2]),2])
mean(smps3[findall(smps3[:,1] .!= smps3[:,2]),2])
quantile(Normal(0,sqrt(8)),0.025)
quantile(smps1[findall(smps1[:,1] .!= smps1[:,2]),2],0.025)
quantile(smps2[findall(smps2[:,1] .!= smps2[:,2]),2],0.025)
quantile(smps3[findall(smps3[:,1] .!= smps3[:,2]),2],0.025)
quantile(smps1[findall(smps1[:,1] .!= smps1[:,2]),2],0.975)
quantile(smps2[findall(smps2[:,1] .!= smps2[:,2]),2],0.975)
quantile(smps3[findall(smps3[:,1] .!= smps3[:,2]),2],0.975)


mean(smps1[findall(smps1[:,1] .!= smps1[:,2]),1] .- smps1[findall(smps1[:,1] .!= smps1[:,2]),2])
mean(smps2[findall(smps2[:,1] .!= smps2[:,2]),1] .- smps2[findall(smps2[:,1] .!= smps2[:,2]),2])
mean(smps3[findall(smps3[:,1] .!= smps3[:,2]),1] .- smps3[findall(smps3[:,1] .!= smps3[:,2]),2])


plot(smps1[findall(smps1[:,1] .!= smps1[:,2]),2])
histogram(smps1[findall(smps1[:,1] .!= smps1[:,2]),2])
mean(smps1[5_000:end,1] .== smps1[5_000:end,2])
mean(smps2[5_000:end,1] .== smps2[5_000:end,2])
mean(smps3[5_000:end,1] .== smps3[5_000:end,2])
mean(smps1[5_000:end,1] .< smps1[5_000:end,2])
mean(smps1[5_000:end,1] .> smps1[5_000:end,2])
mean(smps2[5_000:end,1] .< smps2[5_000:end,2])
mean(smps2[5_000:end,1] .> smps2[5_000:end,2])
mean(smps3[5_000:end,1] .< smps3[5_000:end,2])
mean(smps3[5_000:end,1] .> smps3[5_000:end,2])


plot(collect(1:size(smps1[:,1],1)),cumsum(smps1[:,1] .== smps1[:,2])./collect(1:size(smps1[:,1],1)))
plot!(collect(1:size(smps2[:,1],1)),cumsum(smps2[:,1] .== smps2[:,2])./(1:size(smps2[:,1],1)))
plot(out1["t"][1:100], vec(out1["Sk_x"][:,1,:])[1:100])
plot!(out1["t"][1:100], vec(out1["Sk_x"][:,2,:])[1:100])
plot(out1["t"][1:100], vec(out1["Sk_v"][:,1,:])[1:100])
plot!(out1["t"][1:100], vec(out1["Sk_v"][:,2,:])[1:100])


plot(scatter(quantile(Normal(0,2), collect(0.0001:0.0001:0.9999)),sort(smps1[2:end,1])))
plot(scatter(quantile(Normal(0,2), collect(0.00005:0.00005:0.99995)),sort(smps2[2:end,1])))
plot(scatter(quantile(Normal(0,2), collect(0.000025:0.000025:0.999975)),sort(smps3[2:end,1])))