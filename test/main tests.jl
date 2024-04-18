using DrWatson
@quickactivate "PEM_extrap"
# For src
using DataStructures, LinearAlgebra, Distributions, Random
using Plots

include(srcdir("Sampler.jl"))
include(srcdir("PreProcessing.jl"))
include(srcdir("PostProcessing.jl"))

Random.seed!(123)
n = 100
y = rand(Exponential(1.0),n)
maximum(y)
#breaks = collect(0.05:0.05:(maximum(y) + 0.1))
#breaks = collect(0.05:0.05:1.0)
breaks = collect(1:6)
p = 1
cens = fill(1.0,n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0[2] = 0.1
v0[3] = 0.2
v0[4] = 0.3
v0[5] = 0.4
v0[6] = 0.5
t0 = 0.0
priors = FixedPrior(fill(0.5, size(x0)), 1.0, 1.0, 0.0, 1.0)
nits = 1_000_000
nsmp = 100_000
settings = Settings(nits, nsmp, 0.9, 0.5, 0.0, v0, false)
Random.seed!(23653)
out1 = @time pem_sample(x0, s0, v0, t0, dat, priors, settings)

Random.seed!(3546232)
out2 = @time pem_sample(x0, s0, v0, t0, dat, priors, settings)

Random.seed!(2222)
out3 = @time pem_sample(x0, s0, v0, t0, dat, priors, settings)

smps1 = out1["Smp_x"]
smps2 = out2["Smp_x"]
smps3 = out3["Smp_x"]

mean(smps1[:,2] .== 0.0)
mean(smps2[:,2] .== 0.0)
mean(smps3[:,2] .== 0.0)

mean(smps1[:,1])
mean(smps2[:,1])
mean(smps3[:,1])

mean(smps1[:,2])
mean(smps2[:,2])
mean(smps3[:,2])

mean(smps1[:,2] .== 0.0)
mean(smps2[:,2] .== 0.0)
mean(smps3[:,2] .== 0.0)

mean(smps1[:,3])
mean(smps2[:,3])
mean(smps3[:,3])

mean(smps1[:,3] .== 0.0)
mean(smps2[:,3] .== 0.0)
mean(smps3[:,3] .== 0.0)

quantile(smps1[:,1],0.025)
quantile(smps2[:,1],0.025)
quantile(smps3[:,1],0.025)

quantile(smps1[:,1],0.975)
quantile(smps2[:,1],0.975)
quantile(smps3[:,1],0.975)

mean(smps1[:,2])
mean(smps2[:,2])
mean(smps3[:,2])



mean(smps1[findall(0.0 .!= smps1[:,2]),2])
mean(smps2[findall(0.0 .!= smps2[:,2]),2])
mean(smps3[findall(0.0 .!= smps3[:,2]),2])
quantile(Normal(0,sqrt(1.25)),0.025)
quantile(smps1[findall(0.0 .!= smps1[:,2]),2],0.025)
quantile(smps1[findall(0.0 .!= smps1[:,2]),2],0.975)
quantile(Normal(0,sqrt(1 + 5^2)),0.025)
quantile(smps2[findall(0.0 .!= smps2[:,2]),2],0.025)
quantile(Normal(0,sqrt(2)),0.025)
quantile(smps3[findall(0.0 .!= smps3[:,2]),2],0.025)
quantile(smps1[findall(0.0 .!= smps1[:,2]),2],0.975)
quantile(smps2[findall(0.0 .!= smps2[:,2]),2],0.975)
quantile(smps3[findall(0.0 .!= smps3[:,2]),2],0.975)


mean(smps1[findall(smps1[:,1] .!= smps1[:,2]),1] .- smps1[findall(smps1[:,1] .!= smps1[:,2]),2])
mean(smps2[findall(smps2[:,1] .!= smps2[:,2]),1] .- smps2[findall(smps2[:,1] .!= smps2[:,2]),2])
mean(smps3[findall(smps3[:,1] .!= smps3[:,2]),1] .- smps3[findall(smps3[:,1] .!= smps3[:,2]),2])
histogram(smps1[findall(smps1[:,1] .!= smps1[:,2]),1] .- smps1[findall(smps1[:,1] .!= smps1[:,2]),2])

plot(smps1[findall(smps1[:,1] .!= smps1[:,2]),2])
histogram(smps1[findall(smps1[:,1] .!= smps1[:,2]),2])
mean(smps1[5_000:end,1] .== smps1[5_000:end,2])
mean(smps2[5_000:end,1] .== smps2[5_000:end,2])
mean(smps3[5_000:end,1] .== smps3[5_000:end,2])

mean(smps1[5_000:end,2] .== smps1[5_000:end,3])
mean(smps2[5_000:end,2] .== smps2[5_000:end,3])
mean(smps3[5_000:end,2] .== smps3[5_000:end,3])

mean(smps1[5_000:end,1] .== smps1[5_000:end,3])
mean(smps2[5_000:end,1] .== smps2[5_000:end,3])
mean(smps3[5_000:end,1] .== smps3[5_000:end,3])

mean(smps1[5_000:end,1] .< smps1[5_000:end,2])
mean(smps1[5_000:end,1] .> smps1[5_000:end,2])
mean(smps2[5_000:end,1] .< smps2[5_000:end,2])
mean(smps2[5_000:end,1] .> smps2[5_000:end,2])
mean(smps3[5_000:end,1] .< smps3[5_000:end,2])
mean(smps3[5_000:end,1] .> smps3[5_000:end,2])


plot(collect(1:size(smps1[:,1],1)),cumsum(smps1[:,1] .== smps1[:,2])./collect(1:size(smps1[:,1],1)))
plot!(collect(1:size(smps2[:,1],1)),cumsum(smps2[:,1] .== smps2[:,2])./(1:size(smps2[:,1],1)))
plot!(collect(1:size(smps3[:,1],1)),cumsum(smps3[:,1] .== smps3[:,2])./(1:size(smps3[:,1],1)))
hline!([0.5,0.5])
n_plot = 10_000
n_start = 5000
plot(out1["t"][n_start:n_plot], vec(out1["Sk_x"][:,1,:])[n_start:n_plot])
plot!(out1["t"][n_start:n_plot], vec(out1["Sk_x"][:,2,:])[n_start:n_plot])
plot(out2["t"][n_start:n_plot], vec(out2["Sk_x"][:,1,:])[n_start:n_plot])
plot!(out2["t"][n_start:n_plot], vec(out2["Sk_x"][:,2,:])[n_start:n_plot])
plot(out3["t"][n_start:n_plot], vec(out3["Sk_x"][:,1,:])[n_start:n_plot])
plot!(out3["t"][n_start:n_plot], vec(out3["Sk_x"][:,2,:])[n_start:n_plot])


plot!(out1["t"][1:n_plot], vec(out1["Sk_x"][:,3,:])[1:n_plot])
plot!(out1["t"][1:n_plot], vec(out1["Sk_x"][:,4,:])[1:n_plot])
plot!(out1["t"][1:n_plot], vec(out1["Sk_x"][:,5,:])[1:n_plot])
plot(out1["t"][700:750], vec(out1["Sk_x"][:,1,:])[700:750])
plot!(out1["t"][700:750], vec(out1["Sk_x"][:,2,:])[700:750])

plot(out1["t"][1:25], vec(out1["Sk_x"][:,1,:])[1:25] .- vec(out1["Sk_x"][:,2,:])[1:25])
plot!(out1["t"][1:25], vec(out1["Sk_x"][:,1,:])[1:25])
plot!(out1["t"][1:25], vec(out1["Sk_x"][:,2,:])[1:25])

plot(out1["t"][1:100], vec(out1["Sk_v"][:,1,:])[1:100])
plot!(out1["t"][1:100], vec(out1["Sk_v"][:,2,:])[1:100])


plot(scatter(quantile(Normal(0,1), collect(0.000002:0.000002:0.999998)),sort(smps1[2:end,1])))
plot(scatter(quantile(Normal(0,2), collect(0.00001:0.00001:0.99999)),sort(smps2[2:end,1])))
plot(scatter(quantile(Normal(0,2), collect(0.00001:0.00001:0.99999)),sort(smps3[2:end,1])))

n_plot = 10_000
n_start = 5000
plot(out1["t"][n_start:n_plot], vec(out1["Sk_x"][:,1,:])[n_start:n_plot])
plot!(out1["t"][n_start:n_plot], vec(out1["Sk_x"][:,1,:])[n_start:n_plot] .+ vec(out1["Sk_x"][:,2,:])[n_start:n_plot])
plot!(out1["t"][n_start:n_plot], vec(out1["Sk_x"][:,1,:])[n_start:n_plot] .+ vec(out1["Sk_x"][:,2,:])[n_start:n_plot] .+ vec(out1["Sk_x"][:,3,:])[n_start:n_plot])

plot(out1["t"][n_start:n_plot], vec(out1["Sk_x"][:,1,:])[n_start:n_plot])
plot(out1["t"][n_start:n_plot], vec(out1["Sk_x"][:,2,:])[n_start:n_plot])
plot!(out1["t"][n_start:n_plot], vec(out1["Sk_x"][:,3,:])[n_start:n_plot])


plot(vec(out1["Sk_x"][:,1,:])[n_start:n_plot], vec(out1["Sk_x"][:,2,:])[n_start:n_plot])
plot(vec(out1["Sk_x"][:,2,:])[n_start:n_plot], vec(out1["Sk_x"][:,3,:])[n_start:n_plot])


plot(vec(out1["Sk_x"][:,1,:])[n_start:n_plot], vec(out1["Sk_x"][:,1,:])[n_start:n_plot] .+ vec(out1["Sk_x"][:,2,:])[n_start:n_plot])
plot(vec(out1["Sk_x"][:,1,:])[n_start:n_plot], vec(out1["Sk_x"][:,1,:])[n_start:n_plot] .+ vec(out1["Sk_x"][:,2,:])[n_start:n_plot] .+ vec(out1["Sk_x"][:,3,:])[n_start:n_plot])
plot(vec(out1["Sk_x"][:,1,:])[n_start:n_plot] .+ vec(out1["Sk_x"][:,2,:])[n_start:n_plot], vec(out1["Sk_x"][:,1,:])[n_start:n_plot] .+ vec(out1["Sk_x"][:,2,:])[n_start:n_plot] .+ vec(out1["Sk_x"][:,3,:])[n_start:n_plot])
