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
trunc_ind = findall(y .> 3.0)
y[trunc_ind] .= 3.0
breaks = collect(0.1:0.1:3.5)
p = 1
cens = df.status
cens[trunc_ind] .= 0.0
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
v_abs = vcat(1.0,collect(0.02:0.02:0.68))
x0, v0, s0 = init_params(p, dat, v_abs)
t0 = 0.0
nits = 2_000_000
nsmp = 100_000
settings = Settings(nits, nsmp, 0.9, 0.5, 1.0, v0, false)
Random.seed!(23653)
priors = HyperPrior2(fill(0.4, size(x0)), 0.5, 4.0, 10.0, 0.5, 1.0, 0.0, 1.0)
out1 = @time pem_sample(x0, s0, v0, t0, dat, priors, settings)
priors = HyperPrior2(fill(0.9, size(x0)), 0.5, 9.0, 10.0, 1.0, 1.0, 0.0, 1.0)
out2 = @time pem_sample(x0, s0, v0, t0, dat, priors, settings)
priors = HyperPrior2(fill(0.9, size(x0)), 0.5, 9.0, 10.0, 5.0, 1.0, 0.0, 1.0)
out3 = @time pem_sample(x0, s0, v0, t0, dat, priors, settings)
priors = HyperPrior2(fill(0.9, size(x0)), 0.5, 9.0, 10.0, 5.0, 1.0, 0.0, 1.0)
out4 = @time pem_sample(x0, s0, v0, t0, dat, priors, settings)

plot(out1["Smp_h"])
histogram(out1["Smp_h"])
plot(out2["Smp_h"])
plot(out3["Smp_h"])
plot(out4["Smp_h"])

smps1 = out1["Smp_trans"]
plot_int = 0.02:0.02:3.0
plot_dat1 = survival_plot(plot_int, breaks, vcat(median(exp.(smps1), dims = 2), median(exp.(smps1), dims = 2)[end]), breaks[2] - breaks[1])
plot_dat1l = survival_plot(plot_int, breaks, vcat(exp.(quantile.(eachrow(smps1), 0.025)), exp.(quantile.(eachrow(smps1), 0.025))[end]), breaks[2] - breaks[1])
plot_dat1u = survival_plot(plot_int, breaks, vcat(exp.(quantile.(eachrow(smps1), 0.975)), exp.(quantile.(eachrow(smps1), 0.975))[end]), breaks[2] - breaks[1])
plot(plot_dat1[:,1], plot_dat1[:,2], colour = :red, ylim = (0,1))
plot!(plot_dat1l[:,1], plot_dat1l[:,2], colour = :red, linestyle = :dot)
plot!(plot_dat1u[:,1], plot_dat1u[:,2], colour = :red, linestyle = :dot)

smps1 = out2["Smp_trans"]
plot_dat1 = survival_plot(plot_int, breaks, vcat(median(exp.(smps1), dims = 2), median(exp.(smps1), dims = 2)[end]), breaks[2] - breaks[1])
plot_dat1l = survival_plot(plot_int, breaks, vcat(exp.(quantile.(eachrow(smps1), 0.025)), exp.(quantile.(eachrow(smps1), 0.025))[end]), breaks[2] - breaks[1])
plot_dat1u = survival_plot(plot_int, breaks, vcat(exp.(quantile.(eachrow(smps1), 0.975)), exp.(quantile.(eachrow(smps1), 0.975))[end]), breaks[2] - breaks[1])
plot(plot_dat1[:,1], plot_dat1[:,2], colour = :red, ylim = (0,1))
plot!(plot_dat1l[:,1], plot_dat1l[:,2], colour = :red, linestyle = :dot)
plot!(plot_dat1u[:,1], plot_dat1u[:,2], colour = :red, linestyle = :dot)

smps1 = out3["Smp_trans"]
plot_dat1 = survival_plot(plot_int, breaks, vcat(median(exp.(smps1), dims = 2), median(exp.(smps1), dims = 2)[end]), breaks[2] - breaks[1])
plot_dat1l = survival_plot(plot_int, breaks, vcat(exp.(quantile.(eachrow(smps1), 0.025)), exp.(quantile.(eachrow(smps1), 0.025))[end]), breaks[2] - breaks[1])
plot_dat1u = survival_plot(plot_int, breaks, vcat(exp.(quantile.(eachrow(smps1), 0.975)), exp.(quantile.(eachrow(smps1), 0.975))[end]), breaks[2] - breaks[1])
plot(plot_dat1[:,1], plot_dat1[:,2], colour = :red, ylim = (0,1))
plot!(plot_dat1l[:,1], plot_dat1l[:,2], colour = :red, linestyle = :dot)
plot!(plot_dat1u[:,1], plot_dat1u[:,2], colour = :red, linestyle = :dot)

smps1 = out4["Smp_trans"]
plot_dat1 = survival_plot(plot_int, breaks, vcat(median(exp.(smps1), dims = 2), median(exp.(smps1), dims = 2)[end]), breaks[2] - breaks[1])
plot_dat1l = survival_plot(plot_int, breaks, vcat(exp.(quantile.(eachrow(smps1), 0.025)), exp.(quantile.(eachrow(smps1), 0.025))[end]), breaks[2] - breaks[1])
plot_dat1u = survival_plot(plot_int, breaks, vcat(exp.(quantile.(eachrow(smps1), 0.975)), exp.(quantile.(eachrow(smps1), 0.975))[end]), breaks[2] - breaks[1])
plot(plot_dat1[:,1], plot_dat1[:,2], colour = :red, ylim = (0,1))
plot!(plot_dat1l[:,1], plot_dat1l[:,2], colour = :red, linestyle = :dot)
plot!(plot_dat1u[:,1], plot_dat1u[:,2], colour = :red, linestyle = :dot)

smps1 = out1["Smp_trans"]
plot(vcat(0,breaks), vcat(median(exp.(smps1), dims = 2), median(exp.(smps1), dims = 2)[end]),linetype=:steppost, xlims = (0,3.5), ylim = (0,1))
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(smps1)), 0.025),quantile.(eachrow(exp.(smps1)), 0.025)[end]),linetype=:steppost, xlims = (0,3.5), ylim = (0,3))
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(smps1)), 0.975),quantile.(eachrow(exp.(smps1)), 0.975)[end]),linetype=:steppost, xlims = (0,3.5), ylim = (0,3))

smps1 = out2["Smp_trans"]
plot(vcat(0,breaks), vcat(mean(exp.(smps1), dims = 2), mean(exp.(smps1), dims = 2)[end]),linetype=:steppost, xlims = (0,5), ylim = (0,3))
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(smps1)), 0.025),quantile.(eachrow(exp.(smps1)), 0.025)[end]),linetype=:steppost, xlims = (0,3), ylim = (0,3))
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(smps1)), 0.975),quantile.(eachrow(exp.(smps1)), 0.975)[end]),linetype=:steppost, xlims = (0,3), ylim = (0,3))

smps1 = out3["Smp_trans"]
plot(vcat(0,breaks), vcat(mean(exp.(smps1), dims = 2), mean(exp.(smps1), dims = 2)[end]),linetype=:steppost, xlims = (0,5), ylim = (0,3))
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(smps1)), 0.025),quantile.(eachrow(exp.(smps1)), 0.025)[end]),linetype=:steppost, xlims = (0,3), ylim = (0,3))
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(smps1)), 0.975),quantile.(eachrow(exp.(smps1)), 0.975)[end]),linetype=:steppost, xlims = (0,3), ylim = (0,3))

smps1 = out4["Smp_trans"]
plot(vcat(0,breaks), vcat(mean(exp.(smps1), dims = 2), mean(exp.(smps1), dims = 2)[end]),linetype=:steppost, xlims = (0,5), ylim = (0,3))
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(smps1)), 0.025),quantile.(eachrow(exp.(smps1)), 0.025)[end]),linetype=:steppost, xlims = (0,3), ylim = (0,3))
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(smps1)), 0.975),quantile.(eachrow(exp.(smps1)), 0.975)[end]),linetype=:steppost, xlims = (0,3), ylim = (0,3))

plot(breaks, mean(out1["Smp_s"], dims = 2), ylim = (0,1))
plot!(breaks, mean(out2["Smp_s"], dims = 2), ylim = (0,1))
plot!(breaks, mean(out3["Smp_s"], dims = 2), ylim = (0,1))
plot!(breaks, mean(out4["Smp_s"], dims = 2), ylim = (0,1))

plot(vec(smps1[6,:]),vec(smps1[7,:]))
plot(vec(smps1[6,1:500]),vec(smps1[7,1:500]))
plot(vec(smps1[7,:]),vec(smps1[8,:]))