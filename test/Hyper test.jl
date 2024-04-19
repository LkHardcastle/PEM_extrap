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
#maximum(y)
#breaks = collect(0.05:0.05:(maximum(y) + 0.1))
#breaks = collect(0.05:0.05:1.0)
breaks = collect(0.25:0.25:5.5)
p = 1
cens = fill(1.0,n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
v_abs = vcat(1.0,collect(0.05:0.05:1.05))
x0, v0, s0 = init_params(p, dat, v_abs)
t0 = 0.0
priors = HyperPrior2(fill(0.5, size(x0)), 0.5, 1.0, 1.0, 0.5, 1.0, 0.0, 1.0)
nits = 2_000_000
nsmp = 50_000
settings = Settings(nits, nsmp, 0.9, 0.5, 1.0, v0, false)
Random.seed!(23653)
out1 = @time pem_sample(x0, s0, v0, t0, dat, priors, settings)

smps1 = out1["Smp_trans"]
plot(vcat(0,breaks), vcat(mean(exp.(smps1), dims = 2), mean(exp.(smps1), dims = 2)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(smps1)), 0.025),quantile.(eachrow(exp.(smps1)), 0.025)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(smps1)), 0.975),quantile.(eachrow(exp.(smps1)), 0.975)[end]),linetype=:steppost)
hline!([1,1])

out1["Smp_h"]

plot(out1["Smp_h"])
histogram(out1["Smp_h"])
mean(out1["Smp_h"])
quantile(out1["Smp_h"], 0.025)
quantile(Beta(1,1),0.025)
quantile(out1["Smp_h"], 0.975)
quantile(Beta(1,1),0.975)

Random.seed!(3546232)
priors = FixedPrior(fill(0.5, size(x0)), 0.5, 1.0, 0.0, 1.0)
out2 = @time pem_sample(x0, s0, v0, t0, dat, priors, settings)

smps1 = out2["Smp_trans"]
plot(vcat(0,breaks), vcat(mean(exp.(smps1), dims = 2), mean(exp.(smps1), dims = 2)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(smps1)), 0.025),quantile.(eachrow(exp.(smps1)), 0.025)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(smps1)), 0.975),quantile.(eachrow(exp.(smps1)), 0.975)[end]),linetype=:steppost)
hline!([1,1])

Random.seed!(2222)
priors = FixedPrior(fill(0.8, size(x0)), 0.5, 1.0, 0.0, 1.0)
out3 = @time pem_sample(x0, s0, v0, t0, dat, priors, settings)