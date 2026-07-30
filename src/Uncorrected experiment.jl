using DrWatson
@quickactivate "PEM_extrap"
using DataStructures, LinearAlgebra, Distributions, Random, Optim, Roots, SpecialFunctions, Statistics
using Plots, CSV, DataFrames, Interpolations, MCMCDiagnosticTools, ParetoSmooth

include(srcdir("Sampler.jl"))
include(srcdir("PreProcessing.jl"))
include(srcdir("PostProcessing.jl"))


Random.seed!(2352)
n = 100
y = rand(Weibull(1.0, 0.5), n)
maximum(y)
breaks = vcat(0.01,collect(0.26:0.25:(maximum(y))),maximum(y)+ 0.001)
p = 1
cens = rand(Bernoulli(0.2), n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 20_000
nsmp = 10
test_times = [0.5, 1.5, 2.5]

settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 50.0)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
priors = BasicPrior(1.0, PC(1.0, 2, 0.5, Inf), FixedW([0.5]), 1.0, CtsPois(10.0, 1.0, 100.0, 3.1), [RandomWalk()], [0.1], 2.0)
@time out1 = pem_fit(state0, dat, priors, settings, test_times, 1_000)

settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.005, 100.0)
priors = BasicPrior(1.0, PC(1.0, 2, 0.5, Inf), FixedW([0.5]), 1.0, CtsPois(10.0, 1.0, 100.0, 3.1), [RandomWalk()], [0.1], 2.0)
@time out2 = pem_fit(state0, dat, priors, settings, test_times, 1_000)

settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.001, 500.0)
priors = BasicPrior(1.0, PC(1.0, 2, 0.5, Inf), FixedW([0.5]), 1.0, CtsPois(10.0, 1.0, 100.0, 3.1), [RandomWalk()], [0.1], 2.0)
@time out3 = pem_fit(state0, dat, priors, settings, test_times, 1_000)

settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.0005, 1000.0)
priors = BasicPrior(1.0, PC(1.0, 2, 0.5, Inf), FixedW([0.5]), 1.0, CtsPois(10.0, 1.0, 100.0, 3.1), [RandomWalk()], [0.1], 2.0)
@time out4 = pem_fit(state0, dat, priors, settings, test_times, 1_000)

grid = sort(unique(out1[2]["Sk_s_loc"][cumsum(out1[2]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
grid = grid[1:10:length(grid)]
test_smp = cts_transform(cumsum(out1[2]["Sk_θ"], dims = 2), out1[2]["Sk_s_loc"], grid)
s1 = view(test_smp, 1, :, 10000:20000)

grid = sort(unique(out2[1]["Sk_s_loc"][cumsum(out2[1]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
grid = grid[1:10:length(grid)]
test_smp = cts_transform(cumsum(out2[1]["Sk_θ"], dims = 2), out2[1]["Sk_s_loc"], grid)
s2 = view(test_smp, 1, :, 10000:20000)

grid = sort(unique(out3[1]["Sk_s_loc"][cumsum(out3[1]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
grid = grid[1:10:length(grid)]
test_smp = cts_transform(cumsum(out3[1]["Sk_θ"], dims = 2), out3[1]["Sk_s_loc"], grid)
s3 = view(test_smp, 1, :, 10000:20000)

grid = sort(unique(out4[2]["Sk_s_loc"][cumsum(out4[2]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
grid = grid[1:10:length(grid)]
test_smp = cts_transform(cumsum(out4[2]["Sk_θ"], dims = 2), out4[2]["Sk_s_loc"], grid)
s4 = view(test_smp, 1, :, 10000:20000)

plot(mean(eachcol(s1)), label = "δ = 0.01",color = :blue)
plot!(quantile.(eachrow(s1), 0.05), label = "δ = 0.01", color = :blue, alpha = 0.2, linestyle = :dash)
plot!(quantile.(eachrow(s1), 0.95), label = "δ = 0.01", color = :blue, alpha = 0.2, linestyle = :dash)
plot!(mean(eachcol(s2)), label = "δ = 0.005", color = :green)
plot!(quantile.(eachrow(s2), 0.05), label = "δ = 0.005", color = :green, alpha = 0.2, linestyle = :dash)
plot!(quantile.(eachrow(s2), 0.95), label = "δ = 0.005", color = :green, alpha = 0.2, linestyle = :dash)
plot!(mean(eachcol(s3)), label = "δ = 0.001", color = :red)
plot!(quantile.(eachrow(s3), 0.05), label = "δ = 0.001", color = :red, alpha = 0.2, linestyle = :dash)
plot!(quantile.(eachrow(s3), 0.95), label = "δ = 0.001", color = :red, alpha = 0.2, linestyle = :dash)
plot!(mean(eachcol(s4)), label = "δ = 0.0005", color = :purple)
plot!(quantile.(eachrow(s4), 0.05), label = "δ = 0.0005", color = :purple, alpha = 0.2, linestyle = :dash)
plot!(quantile.(eachrow(s4), 0.95), label = "δ = 0.0005", color = :purple, alpha = 0.2, linestyle = :dash)