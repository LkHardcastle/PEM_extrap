using DrWatson
@quickactivate "PEM_extrap"
# For src
using DataStructures, LinearAlgebra, Distributions, Random
using Plots, CSV, DataFrames

include(srcdir("Sampler.jl"))
include(srcdir("PreProcessing.jl"))
include(srcdir("PostProcessing.jl"))


df = CSV.read(datadir("colon.csv"), DataFrame)
y = df.time./365
maximum(y)
n = length(y)
breaks = collect(0.1:0.1:9)
p = 1
cens = fill(1.0,n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
v_abs = vcat(1.0,collect(0.01:0.01:0.89))
x0, v0, s0 = init_params(p, dat, v_abs)
t0 = 0.0
priors = HyperPrior2(fill(0.5, size(x0)), 0.5, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0)
nits = 2_000_000
nsmp = 50_000
settings = Settings(nits, nsmp, 0.9, 0.5, 1.0, v0, false)
Random.seed!(23653)
out1 = @time pem_sample(x0, s0, v0, t0, dat, priors, settings)
out2 = @time pem_sample(x0, s0, v0, t0, dat, priors, settings)


smps1 = out1["Smp_trans"]
plot(vcat(0,breaks), vcat(mean(exp.(smps1), dims = 2), mean(exp.(smps1), dims = 2)[end]),linetype=:steppost, xlims = (0,5), ylim = (0,1))
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(smps1)), 0.025),quantile.(eachrow(exp.(smps1)), 0.025)[end]),linetype=:steppost, xlims = (0,5), ylim = (0,1))
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(smps1)), 0.975),quantile.(eachrow(exp.(smps1)), 0.975)[end]),linetype=:steppost, xlims = (0,5), ylim = (0,1))


smps1 = out2["Smp_trans"]
plot(vcat(0,breaks), vcat(mean(exp.(smps1), dims = 2), mean(exp.(smps1), dims = 2)[end]),linetype=:steppost, xlims = (0,5), ylim = (0,1))
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(smps1)), 0.025),quantile.(eachrow(exp.(smps1)), 0.025)[end]),linetype=:steppost, xlims = (0,5), ylim = (0,1))
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(smps1)), 0.975),quantile.(eachrow(exp.(smps1)), 0.975)[end]),linetype=:steppost, xlims = (0,5), ylim = (0,1))


smps1 = out2["Smp_trans"]
plot_dat1 = survival_plot(0.02:0.02:5.0, breaks, vcat(median(exp.(smps1), dims = 2), median(exp.(smps1), dims = 2)[end]), breaks[2] - breaks[1])
plot_dat1l = survival_plot(0.02:0.02:5.0, breaks, vcat(exp.(quantile.(eachrow(smps1), 0.025)), exp.(quantile.(eachrow(smps1), 0.025))[end]), breaks[2] - breaks[1])
plot_dat1u = survival_plot(0.02:0.02:5.0, breaks, vcat(exp.(quantile.(eachrow(smps1), 0.975)), exp.(quantile.(eachrow(smps1), 0.975))[end]), breaks[2] - breaks[1])
plot(plot_dat1[:,1], plot_dat1[:,2], colour = :red, ylim = (0,1))
plot!(plot_dat1l[:,1], plot_dat1l[:,2], colour = :red, linestyle = :dot)
plot!(plot_dat1u[:,1], plot_dat1u[:,2], colour = :red, linestyle = :dot)

7.749185370230923e6/7.74918537023091e6

plot(out2["Smp_h"])
plot(out1["Smp_h"])