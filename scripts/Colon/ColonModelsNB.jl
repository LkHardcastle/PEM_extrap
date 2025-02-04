using DrWatson
@quickactivate "PEM_extrap"
# For src
using DataStructures, LinearAlgebra, Distributions, Random, Optim, Roots, SpecialFunctions
using Plots, CSV, DataFrames, RCall, Interpolations

include(srcdir("Sampler.jl"))
include(srcdir("PreProcessing.jl"))
include(srcdir("PostProcessing.jl"))

R"""
library(ggplot2)
library(dplyr)
library(tidyr)
library(cowplot)
cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
"""

Random.seed!(3453)
df = CSV.read(datadir("colon.csv"), DataFrame)
y = df.years
maximum(y)
n = length(y)
breaks = collect(0.03:0.03:3.18)
p = 1
cens = df.status
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 500000
nsmp = 10_000
settings = Settings(nits, nsmp, 1_000_000, 0.2,0.5, 0.5, false, true)

out2["Sk_t"]

plot(out2["Sk_x"][1,3,:])

plot(scatter(log.(out1["Smp_σ"][1,:]), out1["Smp_x"][1,3,:]))
scatter!(log.(out2["Smp_σ"][1,:]), out2["Smp_x"][1,3,:], color = :red)
plot(scatter!(log.(out2["Smp_σ"][1,:]), out2["Smp_x"][1,3,:], color = :red))
plot(out1["Sk_x"][1,3,:])
plot(out2["Sk_x"][1,3,:])


priors1 = BasicPrior(1.0, PC([0.05], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB(5.0, 0.5, 10.0, 150.0, 3.2), [RandomWalk()])
priors2 = BasicPrior(1.0, InvGamma([0.5],[0.01],[0.01]), FixedW([0.5]), 1.0, CtsNB(5.0, 0.5, 10.0, 150.0, 3.2), [RandomWalk()])
priors2 = BasicPrior(1.0, PC([0.05], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB(5.0, 0.5, 10.0, 150.0, 3.2), [GaussLangevin(log(0.2),1.0)])
priors3 = BasicPrior(1.0, PC([0.05], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB(5.0, 0.5, 10.0, 150.0, 3.2), [GammaLangevin(2,7)])
priors4 = BasicPrior(1.0, PC([0.05], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB(5.0, 0.5, 10.0, 150.0, 3.2), [GompertzBaseline(0.5)])

Random.seed!(9102)
@time out1 = pem_sample(state0, dat, priors1, settings)
@time out2 = pem_sample(state0, dat, priors2, settings)
@time out3 = pem_sample(state0, dat, priors3, settings)
@time out4 = pem_sample(state0, dat, priors4, settings)

grid = collect(0.02:0.02:3.05)
breaks_extrap = collect(3.05:0.02:15)
extrap1 = barker_extrapolation(out1, priors1.diff[1], priors1.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)[:,floor(Int,0.5*nsmp):end]
test_smp = cts_transform(cumsum(out1["Smp_x"], dims = 2), out1["Smp_s_loc"], grid)[:,:,floor(Int,0.5*nsmp):end]
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df1 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out2, priors2.diff[1], priors2.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)[:,floor(Int,0.5*nsmp):end]
test_smp = cts_transform(cumsum(out2["Smp_x"], dims = 2), out2["Smp_s_loc"], grid)[:,:,floor(Int,0.5*nsmp):end]
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df2 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out3, priors3.diff[1], priors3.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)[:,floor(Int,0.5*nsmp):end]
test_smp = cts_transform(cumsum(out3["Smp_x"], dims = 2), out3["Smp_s_loc"], grid)[:,:,floor(Int,0.5*nsmp):end]
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df3 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out4, priors4.diff[1], priors4.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)[:,floor(Int,0.5*nsmp):end]
test_smp = cts_transform(cumsum(out4["Smp_x"], dims = 2), out4["Smp_s_loc"], grid)[:,:,floor(Int,0.5*nsmp):end]
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df4 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

CSV.write(datadir("ColonSmps","RW1.csv"), df1)
CSV.write(datadir("ColonSmps","Gaussian1.csv"), df2)
CSV.write(datadir("ColonSmps","Gamma1.csv"), df3)
CSV.write(datadir("ColonSmps","Gompertz1.csv"), df4)

Random.seed!(4362)
@time out1 = pem_sample(state0, dat, priors1, settings)
@time out2 = pem_sample(state0, dat, priors2, settings)
@time out3 = pem_sample(state0, dat, priors3, settings)
@time out4 = pem_sample(state0, dat, priors4, settings)

grid = collect(0.02:0.02:3.05)
breaks_extrap = collect(3.05:0.02:15)
extrap1 = barker_extrapolation(out1, priors1.diff[1], priors1.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)[:,floor(Int,0.5*nsmp):end]
test_smp = cts_transform(cumsum(out1["Smp_x"], dims = 2), out1["Smp_s_loc"], grid)[:,:,floor(Int,0.5*nsmp):end]
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df1 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out2, priors2.diff[1], priors2.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)[:,floor(Int,0.5*nsmp):end]
test_smp = cts_transform(cumsum(out2["Smp_x"], dims = 2), out2["Smp_s_loc"], grid)[:,:,floor(Int,0.5*nsmp):end]
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df2 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out3, priors3.diff[1], priors3.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)[:,floor(Int,0.5*nsmp):end]
test_smp = cts_transform(cumsum(out3["Smp_x"], dims = 2), out3["Smp_s_loc"], grid)[:,:,floor(Int,0.5*nsmp):end]
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df3 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out4, priors4.diff[1], priors4.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)[:,floor(Int,0.5*nsmp):end]
test_smp = cts_transform(cumsum(out4["Smp_x"], dims = 2), out4["Smp_s_loc"], grid)[:,:,floor(Int,0.5*nsmp):end]
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df4 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

CSV.write(datadir("ColonSmps","RW2.csv"), df1)
CSV.write(datadir("ColonSmps","Gaussian2.csv"), df2)
CSV.write(datadir("ColonSmps","Gamma2.csv"), df3)
CSV.write(datadir("ColonSmps","Gompertz2.csv"), df4)