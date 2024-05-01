using DrWatson
@quickactivate "PEM_extrap"
# For src
using DataStructures, LinearAlgebra, Distributions, Random
using Plots, CSV, DataFrames

include(srcdir("Sampler.jl"))
include(srcdir("PreProcessing.jl"))
include(srcdir("PostProcessing.jl"))

df = CSV.read(datadir("colon.csv"), DataFrame)
y = df.years
maximum(y)
n = length(y)
trunc_ind = findall(y .> 3.0)
y[trunc_ind] .= 3.0
breaks = collect(0.5:0.5:3.5)
p = 1
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
#v_abs = vcat(1.0,collect(0.01:0.01:0.6))
v_abs = vcat(1.0,collect(0.1:0.1:0.6))
x0, v0, s0 = init_params(p, dat, v_abs)
t0 = 0.0
nits = 10_000
nsmp = 100
settings = Settings(nits, nsmp, 0.9, 0.5, 1.0, v0, false)

Random.seed!(23653)
priors = HyperPrior2(fill(0.4, size(x0)), 0.5, 20.0, 20.0, 0.1, 1.0, 0.0,1.0,1.0, 0.0)
out1 = @time pem_sample(x0, s0, v0, t0, dat, priors, settings)

Random.seed!(23653)
priors = HyperPrior3(fill(0.4, size(x0)), 0.5, 20.0, 20.0, 
                    ones(1,length(breaks)), 1.0, 
                    1.0,0.0,
                    1.0, ones(1,length(breaks)), ones(1,length(breaks)),
                    0.0)
out2 = @time pem_sample(x0, s0, v0, t0, dat, priors, settings)

smps1 = out1["Smp_trans"]
plot_int = 0.02:0.02:3.0
plot_dat1 = survival_plot(plot_int, breaks, vcat(median(exp.(smps1), dims = 2), median(exp.(smps1), dims = 2)[end]), breaks[2] - breaks[1])
plot_dat1l = survival_plot(plot_int, breaks, vcat(exp.(quantile.(eachrow(smps1), 0.025)), exp.(quantile.(eachrow(smps1), 0.025))[end]), breaks[2] - breaks[1])
plot_dat1u = survival_plot(plot_int, breaks, vcat(exp.(quantile.(eachrow(smps1), 0.975)), exp.(quantile.(eachrow(smps1), 0.975))[end]), breaks[2] - breaks[1])
plot(plot_dat1[:,1], plot_dat1[:,2], colour = :red, ylim = (0,1))
plot!(plot_dat1l[:,1], plot_dat1l[:,2], colour = :red, linestyle = :dot)
plot!(plot_dat1u[:,1], plot_dat1u[:,2], colour = :red, linestyle = :dot)

smps1 = out11["Smp_trans"]
plot_dat1 = survival_plot(plot_int, breaks, vcat(median(exp.(smps1), dims = 2), median(exp.(smps1), dims = 2)[end]), breaks[2] - breaks[1])
plot_dat1l = survival_plot(plot_int, breaks, vcat(exp.(quantile.(eachrow(smps1), 0.025)), exp.(quantile.(eachrow(smps1), 0.025))[end]), breaks[2] - breaks[1])
plot_dat1u = survival_plot(plot_int, breaks, vcat(exp.(quantile.(eachrow(smps1), 0.975)), exp.(quantile.(eachrow(smps1), 0.975))[end]), breaks[2] - breaks[1])
plot!(plot_dat1[:,1], plot_dat1[:,2], colour = :blue, ylim = (0,1))
plot!(plot_dat1l[:,1], plot_dat1l[:,2], colour = :blue, linestyle = :dot)
plot!(plot_dat1u[:,1], plot_dat1u[:,2], colour = :blue, linestyle = :dot)


smps1 = out1["Smp_trans"]
plot(vcat(0,breaks), vcat(median(exp.(smps1), dims = 2), median(exp.(smps1), dims = 2)[end]),linetype=:steppost, xlims = (0,3.5), ylim = (0,1))
#plot(vcat(0,breaks), vcat(mean(exp.(smps1), dims = 2), mean(exp.(smps1), dims = 2)[end]),linetype=:steppost, xlims = (0,3.5), ylim = (0,3))
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(smps1)), 0.025),quantile.(eachrow(exp.(smps1)), 0.025)[end]),linetype=:steppost, xlims = (0,3.5), ylim = (0,1))
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(smps1)), 0.975),quantile.(eachrow(exp.(smps1)), 0.975)[end]),linetype=:steppost, xlims = (0,3.5), ylim = (0,1))


smps1 = out2["Smp_trans"]
plot(vcat(0,breaks), vcat(median(exp.(smps1), dims = 2), median(exp.(smps1), dims = 2)[end]),linetype=:steppost, xlims = (0,3.5), ylim = (0,3))
#plot(vcat(0,breaks), vcat(mean(exp.(smps1), dims = 2), mean(exp.(smps1), dims = 2)[end]),linetype=:steppost, xlims = (0,3.5), ylim = (0,3))
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(smps1)), 0.025),quantile.(eachrow(exp.(smps1)), 0.025)[end]),linetype=:steppost, xlims = (0,3.5), ylim = (0,1))
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(smps1)), 0.975),quantile.(eachrow(exp.(smps1)), 0.975)[end]),linetype=:steppost, xlims = (0,3.5), ylim = (0,1))

smps1 = out1["Smp_trans"]
plot(breaks[1:(end-1)]  .- 0.15, eachcol(smps1[1:(end-1),100:10_000]), legend = false, color = :black, alpha = 0.01)
plot!(breaks[1:(end-1)]  .- 0.15, mean(smps1[1:(end-1),:], dims = 2), colour = :red)
plot!(breaks[1:(end-1)]  .- 0.15, median(smps1[1:(end-1),:], dims = 2), colour = :green)

smps1 = out2["Smp_trans"]
plot(breaks[1:(end-1)] .- 0.15, eachcol(smps1[1:(end-1),100:10_000]), legend = false, color = :black, alpha = 0.01)
plot!(breaks[1:(end-1)]  .- 0.15, mean(smps1[1:(end-1),:], dims = 2), colour = :red)
plot!(breaks[1:(end-1)]  .- 0.15, median(smps1[1:(end-1),:], dims = 2), colour = :green)

smps1 = out1["Smp_x"]
plot(breaks[1:(end-1)] .- 0.15, eachrow(smps1[100:10_000,1:(end-1)]), legend = false, color = :black, alpha = 0.01)
plot!(breaks[1:(end-1)] .- 0.15, vec(mean(smps1[:,1:(end-1)], dims = 1)), colour = :red, ylim = (-7,7))
plot!(breaks[1:(end-1)] .- 0.15, vec(median(smps1[:,1:(end-1)], dims = 1)), colour = :green, ylim = (-7,7))

smps1 = out2["Smp_x"]
plot(breaks[1:(end-1)] .- 0.15, eachrow(smps1[100:10_000,1:(end-1)]), legend = false, color = :black, alpha = 0.01)
plot!(breaks[1:(end-1)] .- 0.15, vec(mean(smps1[:,1:(end-1)], dims = 1)), colour = :red, ylim = (-7,7))
plot!(breaks[1:(end-1)] .- 0.15, vec(median(smps1[:,1:(end-1)], dims = 1)), colour = :green, ylim = (-7,7))