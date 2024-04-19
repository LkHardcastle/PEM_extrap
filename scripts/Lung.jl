using DrWatson
@quickactivate "PEM_extrap"
# For src
using DataStructures, LinearAlgebra, Distributions, Random
using Plots, CSV, DataFrames

include(srcdir("Sampler.jl"))
include(srcdir("PreProcessing.jl"))
include(srcdir("PostProcessing.jl"))


df = CSV.read(datadir("lung.csv"), DataFrame)
y = df.time./365
maximum(y)
n = length(y)
breaks = collect(0.1:0.1:3)
p = 1
cens = fill(1.0,n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
v_abs = vcat(1.0,collect(0.05:0.05:1.45))
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

vcat(0,breaks), vcat(mean(exp.(smps1), dims = 2), mean(exp.(smps1), dims = 2)[end])
plot_dat = survival_plot(0.02:0.02:3.0, breaks, vcat(median(exp.(smps1), dims = 2), median(exp.(smps1), dims = 2)[end]), 0.1)

plot(plot_dat[:,1], plot_dat[:,2])

df = CSV.read(datadir("lung.csv"), DataFrame)
men = findall(df.sex .== 1)
y = (df.time./365)[men]
n = length(y)
maximum(y)
breaks = collect(0.1:0.1:3)
p = 1
cens = fill(1.0,n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
v_abs = vcat(1.0,collect(0.05:0.05:1.45))
x0, v0, s0 = init_params(p, dat, v_abs)
t0 = 0.0
priors = HyperPrior2(fill(0.5, size(x0)), 0.5, 1.0, 1.0, 0.2, 1.0, 0.0, 1.0)
nits = 5_000_000
nsmp = 50_000
settings = Settings(nits, nsmp, 0.9, 0.5, 1.0, v0, false)
Random.seed!(8358)
out2 = @time pem_sample(x0, s0, v0, t0, dat, priors, settings)


smps2 = out2["Smp_trans"]
plot(vcat(0,breaks), vcat(mean(exp.(smps2), dims = 2), mean(exp.(smps2), dims = 2)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(smps2)), 0.025),quantile.(eachrow(exp.(smps2)), 0.025)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(smps2)), 0.975),quantile.(eachrow(exp.(smps2)), 0.975)[end]),linetype=:steppost)

plot_dat = survival_plot(0.02:0.02:3.0, breaks, vcat(median(exp.(smps2), dims = 2), median(exp.(smps2), dims = 2)[end]), 0.1)
plot(plot_dat[:,1], plot_dat[:,2])

df = CSV.read(datadir("lung.csv"), DataFrame)
women = findall(df.sex .== 2)
y = (df.time./365)[women]
n = length(y)
maximum(y)
breaks = collect(0.1:0.1:3)
p = 1
cens = fill(1.0,n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
v_abs = vcat(1.0,collect(0.05:0.05:1.45))
x0, v0, s0 = init_params(p, dat, v_abs)
t0 = 0.0
priors = HyperPrior2(fill(0.5, size(x0)), 0.5, 1.0, 1.0, 0.2, 1.0, 0.0, 1.0)
nits = 5_000_000
nsmp = 50_000
settings = Settings(nits, nsmp, 0.9, 0.5, 1.0, v0, false)
Random.seed!(15235)
out3 = @time pem_sample(x0, s0, v0, t0, dat, priors, settings)


smps3 = out3["Smp_trans"]
plot(vcat(0,breaks), vcat(mean(exp.(smps3), dims = 2), mean(exp.(smps3), dims = 2)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(smps3)), 0.025),quantile.(eachrow(exp.(smps3)), 0.025)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(smps3)), 0.975),quantile.(eachrow(exp.(smps3)), 0.975)[end]),linetype=:steppost)





plot_dat1 = survival_plot(0.02:0.02:3.0, breaks, vcat(median(exp.(smps2), dims = 2), median(exp.(smps2), dims = 2)[end]), 0.1)
plot_dat1l = survival_plot(0.02:0.02:3.0, breaks, vcat(exp.(quantile.(eachrow(smps2), 0.025)), exp.(quantile.(eachrow(smps2), 0.025))[end]), 0.1)
plot_dat1u = survival_plot(0.02:0.02:3.0, breaks, vcat(exp.(quantile.(eachrow(smps2), 0.975)), exp.(quantile.(eachrow(smps2), 0.975))[end]), 0.1)
plot(plot_dat1[:,1], plot_dat1[:,2], colour = :red)
plot!(plot_dat1l[:,1], plot_dat1l[:,2], colour = :red, linestyle = :dot)
plot!(plot_dat1u[:,1], plot_dat1u[:,2], colour = :red, linestyle = :dot)

plot_dat2 = survival_plot(0.02:0.02:3.0, breaks, vcat(median(exp.(smps3), dims = 2), median(exp.(smps3), dims = 2)[end]), 0.1)
plot_dat2l = survival_plot(0.02:0.02:3.0, breaks, vcat(exp.(quantile.(eachrow(smps3), 0.025)), exp.(quantile.(eachrow(smps3), 0.025))[end]), 0.1)
plot_dat2u = survival_plot(0.02:0.02:3.0, breaks, vcat(exp.(quantile.(eachrow(smps3), 0.975)), exp.(quantile.(eachrow(smps3), 0.975))[end]), 0.1)
plot!(plot_dat2[:,1], plot_dat2[:,2], colour = :blue)
plot!(plot_dat2l[:,1], plot_dat2l[:,2], colour = :blue, linestyle = :dot)
plot!(plot_dat2u[:,1], plot_dat2u[:,2], colour = :blue, linestyle = :dot)