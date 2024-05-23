using DrWatson
@quickactivate "PEM_extrap"
# For src
using DataStructures, LinearAlgebra, Distributions, Random, Optim, Roots, SpecialFunctions
using Plots, CSV, DataFrames

include(srcdir("Sampler.jl"))
include(srcdir("PreProcessing.jl"))
include(srcdir("PostProcessing.jl"))

Random.seed!(123)
n = 100
y = rand(Exponential(1.0),n)
breaks = collect(0.5:0.5:ceil(Int,maximum(y)))
p = 1
cens = fill(1.0,n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = BPS(x0, v0, s0, t0, findall(s0))
priors = BasicPrior(1.0, 1.0)
nits = 10_000
nsmp = 10000
settings = Settings(nits, nsmp, 100000, 0.5,0.0, 0.2, false, true)
Random.seed!(123)
@time out1 = pem_sample(state0, dat, priors, settings)
@time out2 = pem_sample(state0, dat, priors, settings)
state0 = ECMC2(x0, v0, s0, t0, true, findall(s0))
@time out3 = pem_sample(state0, dat, priors, settings)
@time out4 = pem_sample(state0, dat, priors, settings)
smps1 = out1["Smp_x"]
smps2 = out2["Smp_x"]
smps3 = out3["Smp_x"]
smps4 = out4["Smp_x"]

smps = copy(smps1)

plot(vec(smps1[:,1,:]),vec(smps1[:,2,:]))
plot(vec(smps1[:,2,:]),vec(smps1[:,12,:]))
plot(vec(smps1[:,3,:]),vec(smps1[:,4,:]))

plot!(vec(smps3[:,1,:]),vec(smps3[:,2,:]))
plot!(vec(smps3[:,2,:]),vec(smps3[:,12,:]))
plot!(vec(smps3[:,3,:]),vec(smps3[:,4,:]))


smps1 = out1["Smp_trans"]
s1 = view(smps1, 1, :, :)
plot(vcat(0,breaks), vcat(mean(exp.(s1), dims = 2), mean(exp.(s1), dims = 2)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.025),quantile.(eachrow(exp.(s1)), 0.025)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.975),quantile.(eachrow(exp.(s1)), 0.975)[end]),linetype=:steppost)

Random.seed!(123)
df = CSV.read(datadir("colon.csv"), DataFrame)
y = df.years
maximum(y)
n = length(y)
breaks = collect(0.1:0.1:3.1)
p = 1
cens = df.status
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
t0 = 0.0
priors = BasicPrior(1.0, 1.0, 0.5, 1.0)
state0 = BPS(x0, v0, s0, t0, findall(s0))

nits = 50000
nsmp = 100000
Random.seed!(123)
settings = Settings(nits, nsmp, 100000, 0.5,0.0, 1.0, false, true)
@time out1 = pem_sample(state0, dat, priors, settings)
settings = Settings(nits, nsmp, 100000, 0.5,0.0, 0.2, false, true)
@time out2 = pem_sample(state0, dat, priors, settings)
settings = Settings(nits, nsmp, 100000, 0.5,0.0, 0.01, false, true)
@time out3 = pem_sample(state0, dat, priors, settings)
state0 = ECMC2(x0, v0, s0, t0, true, findall(s0))
settings = Settings(nits, nsmp, 100000, 0.5,0.0, 1.0, false, true)
@time out4 = pem_sample(state0, dat, priors, settings)
settings = Settings(nits, nsmp, 100000, 0.5,0.0, 0.2, false, true)
@time out5 = pem_sample(state0, dat, priors, settings)
settings = Settings(nits, nsmp, 100000, 0.5,0.0, 0.01, false, true)
@time out6 = pem_sample(state0, dat, priors, settings)

smps1 = out1["Smp_trans"]
s1 = view(smps1, 1, :, :)
plot(vcat(0,breaks), vcat(mean(exp.(s1), dims = 2), mean(exp.(s1), dims = 2)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.025),quantile.(eachrow(exp.(s1)), 0.025)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.975),quantile.(eachrow(exp.(s1)), 0.975)[end]),linetype=:steppost, ylim = (0,1))

smps1 = out2["Smp_trans"]
s1 = view(smps1, 1, :, :)
plot(vcat(0,breaks), vcat(mean(exp.(s1), dims = 2), mean(exp.(s1), dims = 2)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.025),quantile.(eachrow(exp.(s1)), 0.025)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.975),quantile.(eachrow(exp.(s1)), 0.975)[end]),linetype=:steppost, ylim = (0,1))

smps1 = out3["Smp_trans"]
s1 = view(smps1, 1, :, :)
plot(vcat(0,breaks), vcat(mean(exp.(s1), dims = 2), mean(exp.(s1), dims = 2)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.025),quantile.(eachrow(exp.(s1)), 0.025)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.975),quantile.(eachrow(exp.(s1)), 0.975)[end]),linetype=:steppost, ylim = (0,1))

smps1 = out4["Smp_trans"]
s1 = view(smps1, 1, :, :)
plot(vcat(0,breaks), vcat(mean(exp.(s1), dims = 2), mean(exp.(s1), dims = 2)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.025),quantile.(eachrow(exp.(s1)), 0.025)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.975),quantile.(eachrow(exp.(s1)), 0.975)[end]),linetype=:steppost, ylim = (0,1))

smps1 = out5["Smp_trans"]
s1 = view(smps1, 1, :, :)
plot(vcat(0,breaks), vcat(mean(exp.(s1), dims = 2), mean(exp.(s1), dims = 2)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.025),quantile.(eachrow(exp.(s1)), 0.025)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.975),quantile.(eachrow(exp.(s1)), 0.975)[end]),linetype=:steppost, ylim = (0,1))

smps1 = out6["Smp_trans"]
s1 = view(smps1, 1, :, :)
plot(vcat(0,breaks), vcat(mean(exp.(s1), dims = 2), mean(exp.(s1), dims = 2)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.025),quantile.(eachrow(exp.(s1)), 0.025)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.975),quantile.(eachrow(exp.(s1)), 0.975)[end]),linetype=:steppost, ylim = (0,1))


Random.seed!(123)
df = CSV.read(datadir("colon.csv"), DataFrame)
y = df.years
maximum(y)
n = length(y)
breaks = collect(0.1:0.1:3.5)
p = 1
cens = df.status
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, t0, true, findall(s0))
nits = 50000
nsmp = 100000
settings = Settings(nits, nsmp, 100000, 0.5,0.0, 0.2, false, true)
Random.seed!(123)
priors = BasicPrior(1.0, Cauchy(0.1), Beta(0.5), 1.0)
@time out1 = pem_sample(state0, dat, priors, settings)
priors = BasicPrior(1.0, Cauchy(0.5), Beta(0.5), 1.0)
@time out2 = pem_sample(state0, dat, priors, settings)
priors = BasicPrior(1.0, Cauchy(1.0), Beta(0.5), 1.0)
@time out3 = pem_sample(state0, dat, priors, settings)


smps1 = out1["Smp_trans"]
s1 = view(smps1, 1, :, :)
plot(vcat(0,breaks), vcat(mean(exp.(s1), dims = 2), mean(exp.(s1), dims = 2)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.025),quantile.(eachrow(exp.(s1)), 0.025)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.975),quantile.(eachrow(exp.(s1)), 0.975)[end]),linetype=:steppost, ylim = (0,1))

smps1 = out2["Smp_trans"]
s1 = view(smps1, 1, :, :)
plot(vcat(0,breaks), vcat(mean(exp.(s1), dims = 2), mean(exp.(s1), dims = 2)[end]),linetype=:steppost)
plot!(vcat(0,breaks), vcat(mean(exp.(s1), dims = 2), median(exp.(s1), dims = 2)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.025),quantile.(eachrow(exp.(s1)), 0.025)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.975),quantile.(eachrow(exp.(s1)), 0.975)[end]),linetype=:steppost, ylim = (0,1))

smps1 = out3["Smp_trans"]
s1 = view(smps1, 1, :, :)
plot(vcat(0,breaks), vcat(mean(exp.(s1), dims = 2), mean(exp.(s1), dims = 2)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.025),quantile.(eachrow(exp.(s1)), 0.025)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.975),quantile.(eachrow(exp.(s1)), 0.975)[end]),linetype=:steppost, ylim = (0,1))
