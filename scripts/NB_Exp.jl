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
nits = 200_000
nsmp = 10_000
settings = Settings(nits, nsmp, 1_000_000, 1.0, 2.0, 0.5, false, true)

priors1 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPois(10.0, 150.0, 3.2), [RandomWalk()])
priors2 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB(10.0, 1.0, 10.0, 150.0, 3.2), [RandomWalk()])
priors3 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB(5.0, 0.5, 10.0, 150.0, 3.2), [RandomWalk()])
priors4 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB(2.5, 0.25, 10.0, 150.0, 3.2), [RandomWalk()])

Random.seed!(24562)
@time out1 = pem_sample(state0, dat, priors1, settings)
@time out2 = pem_sample(state0, dat, priors2, settings)
@time out3 = pem_sample(state0, dat, priors3, settings)
@time out4 = pem_sample(state0, dat, priors4, settings)

Random.seed!(1237)
grid = sort(unique(out1["Smp_s_loc"][cumsum(out1["Smp_s"],dims = 1)[1,:,:] .> 0.0]))
grid = grid[1:10:length(grid)]
breaks_extrap = collect(3.2:0.02:15)
extrap1 = barker_extrapolation(out1, priors1.diff[1], priors1.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out1["Smp_θ"], dims = 2), out1["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df9 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out2, priors2.diff[1], priors2.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out2["Smp_θ"], dims = 2), out2["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df10 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out3, priors3.diff[1], priors3.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out3["Smp_θ"], dims = 2), out3["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df11 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out4, priors4.diff[1], priors4.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out4["Smp_θ"], dims = 2), out4["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df12 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

CSV.write(datadir("ColonSmps","BetaNB11.csv"), df9)
CSV.write(datadir("ColonSmps","BetaNB12.csv"), df10)
CSV.write(datadir("ColonSmps","BetaNB13.csv"), df11)
CSV.write(datadir("ColonSmps","BetaNB14.csv"), df12)

priors1 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPois(5.0, 150.0, 3.2), [RandomWalk()])
priors2 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB(5.0, 1.0, 10.0, 150.0, 3.2), [RandomWalk()])
priors3 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB(2.5, 0.5, 10.0, 150.0, 3.2), [RandomWalk()])
priors4 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB(1.25, 0.25, 10.0, 150.0, 3.2), [RandomWalk()])
Random.seed!(24562)
@time out1 = pem_sample(state0, dat, priors1, settings)
@time out2 = pem_sample(state0, dat, priors2, settings)
@time out3 = pem_sample(state0, dat, priors3, settings)
@time out4 = pem_sample(state0, dat, priors4, settings)

Random.seed!(1237)
grid = sort(unique(out1["Smp_s_loc"][cumsum(out1["Smp_s"],dims = 1)[1,:,:] .> 0.0]))
grid = grid[1:10:length(grid)]
breaks_extrap = collect(3.2:0.02:15)
extrap1 = barker_extrapolation(out1, priors1.diff[1], priors1.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out1["Smp_θ"], dims = 2), out1["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df9 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out2, priors2.diff[1], priors2.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out2["Smp_θ"], dims = 2), out2["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df10 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out3, priors3.diff[1], priors3.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out3["Smp_θ"], dims = 2), out3["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df11 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out4, priors4.diff[1], priors4.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out4["Smp_θ"], dims = 2), out4["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df12 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

CSV.write(datadir("ColonSmps","BetaNB21.csv"), df9)
CSV.write(datadir("ColonSmps","BetaNB22.csv"), df10)
CSV.write(datadir("ColonSmps","BetaNB23.csv"), df11)
CSV.write(datadir("ColonSmps","BetaNB24.csv"), df12)

priors1 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPois(20.0, 150.0, 3.2), [RandomWalk()])
priors2 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB(20.0, 1.0, 10.0, 150.0, 3.2), [RandomWalk()])
priors3 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB(10.0, 0.5, 10.0, 150.0, 3.2), [RandomWalk()])
priors4 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB(5.0, 0.25, 10.0, 150.0, 3.2), [RandomWalk()])

Random.seed!(24562)
@time out1 = pem_sample(state0, dat, priors1, settings)
@time out2 = pem_sample(state0, dat, priors2, settings)
@time out3 = pem_sample(state0, dat, priors3, settings)
@time out4 = pem_sample(state0, dat, priors4, settings)

Random.seed!(1237)
grid = sort(unique(out1["Smp_s_loc"][cumsum(out1["Smp_s"],dims = 1)[1,:,:] .> 0.0]))
grid = grid[1:10:length(grid)]
breaks_extrap = collect(3.2:0.02:15)
extrap1 = barker_extrapolation(out1, priors1.diff[1], priors1.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out1["Smp_θ"], dims = 2), out1["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df9 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out2, priors2.diff[1], priors2.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out2["Smp_θ"], dims = 2), out2["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df10 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out3, priors3.diff[1], priors3.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out3["Smp_θ"], dims = 2), out3["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df11 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out4, priors4.diff[1], priors4.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out4["Smp_θ"], dims = 2), out4["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df12 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

CSV.write(datadir("ColonSmps","BetaNB31.csv"), df9)
CSV.write(datadir("ColonSmps","BetaNB32.csv"), df10)
CSV.write(datadir("ColonSmps","BetaNB33.csv"), df11)
CSV.write(datadir("ColonSmps","BetaNB34.csv"), df12)


priors1 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPois(1.0, 150.0, 3.2), [RandomWalk()])
priors2 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB(0.5, 0.0, 10.0, 150.0, 3.2), [RandomWalk()])
priors3 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB(1/3, 0.0, 10.0, 150.0, 3.2), [RandomWalk()])
priors4 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB(0.01, 0.01, 10.0, 150.0, 3.2), [RandomWalk()])
Random.seed!(24562)
@time out1 = pem_sample(state0, dat, priors1, settings)
@time out2 = pem_sample(state0, dat, priors2, settings)
@time out3 = pem_sample(state0, dat, priors3, settings)
@time out4 = pem_sample(state0, dat, priors4, settings)

Random.seed!(1237)
grid = sort(unique(out1["Smp_s_loc"][cumsum(out1["Smp_s"],dims = 1)[1,:,:] .> 0.0]))
grid = grid[1:10:length(grid)]
breaks_extrap = collect(3.2:0.02:15)
extrap1 = barker_extrapolation(out1, priors1.diff[1], priors1.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out1["Smp_θ"], dims = 2), out1["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df9 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out2, priors2.diff[1], priors2.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out2["Smp_θ"], dims = 2), out2["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df10 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out3, priors3.diff[1], priors3.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out3["Smp_θ"], dims = 2), out3["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df11 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out4, priors4.diff[1], priors4.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out4["Smp_θ"], dims = 2), out4["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df12 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

CSV.write(datadir("ColonSmps","BetaNB41.csv"), df9)
CSV.write(datadir("ColonSmps","BetaNB42.csv"), df10)
CSV.write(datadir("ColonSmps","BetaNB43.csv"), df11)
CSV.write(datadir("ColonSmps","BetaNB44.csv"), df12)


priors1 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNBmix(5.0, .5, 1.0, 1.0, 1.0, 10.0, 150.0, 3.2), [RandomWalk()])
priors2 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNBmix(5.0, .5, 20.0, 1.0, 1.0, 10.0, 150.0, 3.2), [RandomWalk()])
priors3 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNBmix(5.0, .5, .01, .01, 1.0, 10.0, 150.0, 3.2), [RandomWalk()])
priors4 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNBmix(5.0, .5, 10.0, 1.0, 1.0, 10.0, 150.0, 3.2), [RandomWalk()])
Random.seed!(24562)
@time out1 = pem_sample(state0, dat, priors1, settings)
@time out2 = pem_sample(state0, dat, priors2, settings)
@time out3 = pem_sample(state0, dat, priors3, settings)
@time out4 = pem_sample(state0, dat, priors4, settings)
plot(out1["Sk_γ"][1:1_000])
plot(out2["Sk_γ"][1:1_000])
plot(out3["Sk_γ"][1:1_000])
plot(out4["Sk_γ"][1:1_000])

Random.seed!(1237)
grid = sort(unique(out1["Smp_s_loc"][cumsum(out1["Smp_s"],dims = 1)[1,:,:] .> 0.0]))
grid = grid[1:10:length(grid)]
breaks_extrap = collect(3.2:0.02:15)
extrap1 = barker_extrapolation(out1, priors1.diff[1], priors1.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out1["Smp_θ"], dims = 2), out1["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df9 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out2, priors2.diff[1], priors2.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out2["Smp_θ"], dims = 2), out2["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df10 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out3, priors3.diff[1], priors3.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out3["Smp_θ"], dims = 2), out3["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df11 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out4, priors4.diff[1], priors4.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out4["Smp_θ"], dims = 2), out4["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df12 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

CSV.write(datadir("ColonSmps","BetaNBmix1.csv"), df9)
CSV.write(datadir("ColonSmps","BetaNBmix2.csv"), df10)
CSV.write(datadir("ColonSmps","BetaNBmix3.csv"), df11)
CSV.write(datadir("ColonSmps","BetaNBmix4.csv"), df12)

priors1 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNBmix(5.0, .5, 1.0, 1.0, 1.0, 10.0, 150.0, 3.2), [RandomWalk()])
priors2 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNBmix(1.0, 1.0, 5.0, .5, 1.0, 10.0, 150.0, 3.2), [RandomWalk()])
priors3 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNBmix(20.0, 1.0, .01, .01, 1.0, 10.0, 150.0, 3.2), [RandomWalk()])
priors4 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNBmix(10.0, .5, .5, .5, 1.0, 10.0, 150.0, 3.2), [RandomWalk()])
Random.seed!(24562)
@time out1 = pem_sample(state0, dat, priors1, settings)
@time out2 = pem_sample(state0, dat, priors2, settings)
@time out3 = pem_sample(state0, dat, priors3, settings)
@time out4 = pem_sample(state0, dat, priors4, settings)

Random.seed!(1237)
grid = sort(unique(out1["Smp_s_loc"][cumsum(out1["Smp_s"],dims = 1)[1,:,:] .> 0.0]))
grid = grid[1:10:length(grid)]
breaks_extrap = collect(3.2:0.02:15)
extrap1 = barker_extrapolation(out1, priors1.diff[1], priors1.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out1["Smp_θ"], dims = 2), out1["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df9 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out2, priors2.diff[1], priors2.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out2["Smp_θ"], dims = 2), out2["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df10 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out3, priors3.diff[1], priors3.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out3["Smp_θ"], dims = 2), out3["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df11 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out4, priors4.diff[1], priors4.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out4["Smp_θ"], dims = 2), out4["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df12 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

CSV.write(datadir("ColonSmps","BetaNBmix5.csv"), df9)
CSV.write(datadir("ColonSmps","BetaNBmix6.csv"), df10)
CSV.write(datadir("ColonSmps","BetaNBmix7.csv"), df11)
CSV.write(datadir("ColonSmps","BetaNBmix8.csv"), df12)

priors1 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPoisRE(5.0, 5.0, 5.0, 1.0, 0.5, 150.0, 3.2), [RandomWalk()])
priors2 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPoisRE(5.0, 5.0, 1.0, 1.0, 0.5, 150.0, 3.2), [RandomWalk()])
priors3 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPoisRE(5.0, 0.0, 5.0, 1.0, 0.5, 150.0, 3.2), [RandomWalk()])
priors4 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPoisRE(5.0, 0.0, 1.0, 1.0, 0.5, 150.0, 3.2), [RandomWalk()])
Random.seed!(24562)
@time out1 = pem_sample(state0, dat, priors1, settings)
@time out2 = pem_sample(state0, dat, priors2, settings)
@time out3 = pem_sample(state0, dat, priors3, settings)
@time out4 = pem_sample(state0, dat, priors4, settings)

Random.seed!(1237)
grid = sort(unique(out1["Smp_s_loc"][cumsum(out1["Smp_s"],dims = 1)[1,:,:] .> 0.0]))
grid = grid[1:10:length(grid)]
breaks_extrap = collect(3.2:0.02:15)
extrap1 = barker_extrapolation(out1, priors1.diff[1], priors1.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out1["Smp_θ"], dims = 2), out1["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df9 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out2, priors2.diff[1], priors2.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out2["Smp_θ"], dims = 2), out2["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df10 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out3, priors3.diff[1], priors3.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out3["Smp_θ"], dims = 2), out3["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df11 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out4, priors4.diff[1], priors4.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out4["Smp_θ"], dims = 2), out4["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df12 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

CSV.write(datadir("ColonSmps","PoisRE1.csv"), df9)
CSV.write(datadir("ColonSmps","PoisRE2.csv"), df10)
CSV.write(datadir("ColonSmps","PoisRE3.csv"), df11)
CSV.write(datadir("ColonSmps","PoisRE4.csv"), df12)


priors1 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPoisRE(5.0, 3.0, 1.0, 1.0, 0.5, 150.0, 3.2), [RandomWalk()])
priors2 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPoisRE(5.0, 3.0, 2.0, 1.0, 0.5, 150.0, 3.2), [RandomWalk()])
priors3 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPoisRE(5.0, 3.0, 5.0, 1.0, 0.5, 150.0, 3.2), [RandomWalk()])
priors4 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPoisRE(5.0, 3.0, 10.0, 1.0, 0.5, 150.0, 3.2), [RandomWalk()])
Random.seed!(24562)
@time out1 = pem_sample(state0, dat, priors1, settings)
@time out2 = pem_sample(state0, dat, priors2, settings)
@time out3 = pem_sample(state0, dat, priors3, settings)
@time out4 = pem_sample(state0, dat, priors4, settings)

Random.seed!(1237)
grid = sort(unique(out1["Smp_s_loc"][cumsum(out1["Smp_s"],dims = 1)[1,:,:] .> 0.0]))
grid = grid[1:10:length(grid)]
breaks_extrap = collect(3.2:0.02:15)
extrap1 = barker_extrapolation(out1, priors1.diff[1], priors1.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out1["Smp_θ"], dims = 2), out1["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df9 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out2, priors2.diff[1], priors2.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out2["Smp_θ"], dims = 2), out2["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df10 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out3, priors3.diff[1], priors3.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out3["Smp_θ"], dims = 2), out3["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df11 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out4, priors4.diff[1], priors4.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out4["Smp_θ"], dims = 2), out4["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df12 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

CSV.write(datadir("ColonSmps","PoisRE5.csv"), df9)
CSV.write(datadir("ColonSmps","PoisRE6.csv"), df10)
CSV.write(datadir("ColonSmps","PoisRE7.csv"), df11)
CSV.write(datadir("ColonSmps","PoisRE8.csv"), df12)


priors1 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB2(1.0, 1.0, 0.1, 0.1, 5.0, 1.0, 150.0, 3.2), [RandomWalk()])
priors2 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB2(1.0, 1.0, 1.0, 1.0, 5.0, 1.0, 150.0, 3.2), [RandomWalk()])
priors3 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB2(5.0, 1.0, 0.1, 0.1, 5.0, 1.0, 150.0, 3.2), [RandomWalk()])
priors4 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB2(5.0, 1.0, 1.0, 1.0, 5.0, 1.0, 150.0, 3.2), [RandomWalk()])
Random.seed!(24562)
@time out1 = pem_sample(state0, dat, priors1, settings)
@time out2 = pem_sample(state0, dat, priors2, settings)
@time out3 = pem_sample(state0, dat, priors3, settings)
@time out4 = pem_sample(state0, dat, priors4, settings)

Random.seed!(1237)
grid = sort(unique(out1["Smp_s_loc"][cumsum(out1["Smp_s"],dims = 1)[1,:,:] .> 0.0]))
grid = grid[1:10:length(grid)]
breaks_extrap = collect(3.2:0.02:15)
extrap1 = barker_extrapolation(out1, priors1.diff[1], priors1.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out1["Smp_θ"], dims = 2), out1["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df9 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out2, priors2.diff[1], priors2.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out2["Smp_θ"], dims = 2), out2["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df10 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out3, priors3.diff[1], priors3.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out3["Smp_θ"], dims = 2), out3["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df11 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out4, priors4.diff[1], priors4.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out4["Smp_θ"], dims = 2), out4["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df12 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

CSV.write(datadir("ColonSmps","NB21.csv"), df9)
CSV.write(datadir("ColonSmps","NB22.csv"), df10)
CSV.write(datadir("ColonSmps","NB23.csv"), df11)
CSV.write(datadir("ColonSmps","NB24.csv"), df12)


priors1 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB2(20.0, 1.0, 0.1, 0.1, 5.0, 1.0, 150.0, 3.2), [RandomWalk()])
priors2 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB2(10.0, 1.0, 0.1, 0.1, 5.0, 1.0, 150.0, 3.2), [RandomWalk()])
priors3 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB2(5.0, 1.0, 0.1, 0.1, 5.0, 1.0, 150.0, 3.2), [RandomWalk()])
priors4 = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB2(1.0, 1.0, 0.1, 0.1, 5.0, 1.0, 150.0, 3.2), [RandomWalk()])
Random.seed!(24562)
@time out1 = pem_sample(state0, dat, priors1, settings)
@time out2 = pem_sample(state0, dat, priors2, settings)
@time out3 = pem_sample(state0, dat, priors3, settings)
@time out4 = pem_sample(state0, dat, priors4, settings)

Random.seed!(1237)
grid = sort(unique(out1["Smp_s_loc"][cumsum(out1["Smp_s"],dims = 1)[1,:,:] .> 0.0]))
grid = grid[1:10:length(grid)]
breaks_extrap = collect(3.2:0.02:15)
extrap1 = barker_extrapolation(out1, priors1.diff[1], priors1.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out1["Smp_θ"], dims = 2), out1["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df9 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out2, priors2.diff[1], priors2.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out2["Smp_θ"], dims = 2), out2["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df10 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out3, priors3.diff[1], priors3.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out3["Smp_θ"], dims = 2), out3["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df11 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out4, priors4.diff[1], priors4.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out4["Smp_θ"], dims = 2), out4["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df12 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

CSV.write(datadir("ColonSmps","NB25.csv"), df9)
CSV.write(datadir("ColonSmps","NB26.csv"), df10)
CSV.write(datadir("ColonSmps","NB27.csv"), df11)
CSV.write(datadir("ColonSmps","NB28.csv"), df12)


df1 = CSV.read(datadir("ColonSmps","BetaNB11.csv"), DataFrame)
df2 = CSV.read(datadir("ColonSmps","BetaNB12.csv"), DataFrame)
df3 = CSV.read(datadir("ColonSmps","BetaNB13.csv"), DataFrame)
df4 = CSV.read(datadir("ColonSmps","BetaNB14.csv"), DataFrame)
df5 = CSV.read(datadir("ColonSmps","BetaNB21.csv"), DataFrame)
df6 = CSV.read(datadir("ColonSmps","BetaNB22.csv"), DataFrame)
df7 = CSV.read(datadir("ColonSmps","BetaNB23.csv"), DataFrame)
df8 = CSV.read(datadir("ColonSmps","BetaNB24.csv"), DataFrame)
df9 = CSV.read(datadir("ColonSmps","BetaNB31.csv"), DataFrame)
df10 = CSV.read(datadir("ColonSmps","BetaNB32.csv"), DataFrame)
df11 = CSV.read(datadir("ColonSmps","BetaNB33.csv"), DataFrame)
df12 = CSV.read(datadir("ColonSmps","BetaNB34.csv"), DataFrame)
df13 = CSV.read(datadir("ColonSmps","BetaNB41.csv"), DataFrame)
df14 = CSV.read(datadir("ColonSmps","BetaNB42.csv"), DataFrame)
df15 = CSV.read(datadir("ColonSmps","BetaNB43.csv"), DataFrame)
df16 = CSV.read(datadir("ColonSmps","BetaNB44.csv"), DataFrame)
df17 = CSV.read(datadir("ColonSmps","BetaNBmix1.csv"), DataFrame)
df18 = CSV.read(datadir("ColonSmps","BetaNBmix2.csv"), DataFrame)
df19 = CSV.read(datadir("ColonSmps","BetaNBmix3.csv"), DataFrame)
df20 = CSV.read(datadir("ColonSmps","BetaNBmix4.csv"), DataFrame)
df21 = CSV.read(datadir("ColonSmps","BetaNBmix5.csv"), DataFrame)
df22 = CSV.read(datadir("ColonSmps","BetaNBmix6.csv"), DataFrame)
df23 = CSV.read(datadir("ColonSmps","BetaNBmix7.csv"), DataFrame)
df24 = CSV.read(datadir("ColonSmps","BetaNBmix8.csv"), DataFrame)
df25 = CSV.read(datadir("ColonSmps","PoisRE1.csv"), DataFrame)
df26 = CSV.read(datadir("ColonSmps","PoisRE2.csv"), DataFrame)
df27 = CSV.read(datadir("ColonSmps","PoisRE3.csv"), DataFrame)
df28 = CSV.read(datadir("ColonSmps","PoisRE4.csv"), DataFrame)
df29 = CSV.read(datadir("ColonSmps","PoisRE5.csv"), DataFrame)
df30 = CSV.read(datadir("ColonSmps","PoisRE6.csv"), DataFrame)
df31 = CSV.read(datadir("ColonSmps","PoisRE7.csv"), DataFrame)
df32 = CSV.read(datadir("ColonSmps","PoisRE8.csv"), DataFrame)
df33 = CSV.read(datadir("ColonSmps","NB21.csv"), DataFrame)
df34 = CSV.read(datadir("ColonSmps","NB22.csv"), DataFrame)
df35 = CSV.read(datadir("ColonSmps","NB23.csv"), DataFrame)
df36 = CSV.read(datadir("ColonSmps","NB24.csv"), DataFrame)
df37 = CSV.read(datadir("ColonSmps","NB25.csv"), DataFrame)
df38 = CSV.read(datadir("ColonSmps","NB26.csv"), DataFrame)
df39 = CSV.read(datadir("ColonSmps","NB27.csv"), DataFrame)
df40 = CSV.read(datadir("ColonSmps","NB28.csv"), DataFrame)

R"""
dat1 = data.frame($df1)
dat1 = cbind(dat1, "P - 10")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat2 = data.frame($df2)
dat2 = cbind(dat2, "NB - 10, 1")
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat3 = data.frame($df3)
dat3 = cbind(dat3, "NB - 5, .5")
colnames(dat3) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat4 = data.frame($df4)
dat4 = cbind(dat4, "NB - 2.5, .25")
colnames(dat4) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat_1 <- rbind(dat1, dat2, dat3, dat4)
"""

R"""
dat1 = data.frame($df5)
dat1 = cbind(dat1, "P - 5")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat2 = data.frame($df6)
dat2 = cbind(dat2, "NB - 5, 1")
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat3 = data.frame($df7)
dat3 = cbind(dat3, "NB - 2.5, .5")
colnames(dat3) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat4 = data.frame($df8)
dat4 = cbind(dat4, "NB - 1.25, .25")
colnames(dat4) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat_2 <- rbind(dat1, dat2, dat3, dat4)
"""

R"""
dat1 = data.frame($df9)
dat1 = cbind(dat1, "P - 20")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat2 = data.frame($df10)
dat2 = cbind(dat2, "NB - 20, 1")
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat3 = data.frame($df11)
dat3 = cbind(dat3, "NB - 10, .5")
colnames(dat3) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat4 = data.frame($df12)
dat4 = cbind(dat4, "NB - 5, .25")
colnames(dat4) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat_3 <- rbind(dat1, dat2, dat3, dat4)
"""

R"""
dat1 = data.frame($df13)
dat1 = cbind(dat1, "P - 1")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat2 = data.frame($df14)
dat2 = cbind(dat2, "NB - Jeffrey's")
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat3 = data.frame($df15)
dat3 = cbind(dat3, "NB - Neutral")
colnames(dat3) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat4 = data.frame($df16)
dat4 = cbind(dat4, "NB - Non-Inf")
colnames(dat4) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat_4 <- rbind(dat1, dat2, dat3, dat4)
"""

R"""
dat1 = data.frame($df4)
dat1 = cbind(dat1, "NB - (10) 2.5, .25")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat2 = data.frame($df8)
dat2 = cbind(dat2, "NB - (5) 1.25, .25")
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat3 = data.frame($df12)
dat3 = cbind(dat3, "NB - (20) 5, .25")
colnames(dat3) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat4 = data.frame($df16)
dat4 = cbind(dat4, "NB - (1) .01, .01")
colnames(dat4) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat_all <- rbind(dat1, dat2, dat3, dat4)
"""

R"""
dat1 = data.frame($df1)
dat1 = cbind(dat1, "P - 10")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat2 = data.frame($df5)
dat2 = cbind(dat2, "P - 5")
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat3 = data.frame($df9)
dat3 = cbind(dat3, "P - 20")
colnames(dat3) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat4 = data.frame($df13)
dat4 = cbind(dat4, "P - 1")
colnames(dat4) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat_p <- rbind(dat1, dat2, dat3, dat4)
"""

R"""
dat1 = data.frame($df17)
dat1 = cbind(dat1, "Mix - (5.0, .5, 1.0, 1.0)")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat2 = data.frame($df18)
dat2 = cbind(dat2, "Mix - (5.0, .5, 20.0, 1.0)")
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat3 = data.frame($df19)
dat3 = cbind(dat3, "Mix - (5.0, .5, .01, .01)")
colnames(dat3) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat4 = data.frame($df20)
dat4 = cbind(dat4, "Mix - (5.0, .5, 10.0, 1.0)")
colnames(dat4) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat5 = data.frame($df21)
dat5 = cbind(dat5, "Mix - (5.0, .5, 1.0, 1.0)2")
colnames(dat5) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat6 = data.frame($df22)
dat6 = cbind(dat6, "Mix - (1.0, 1.0, 5.0, .5)")
colnames(dat6) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat7 = data.frame($df23)
dat7 = cbind(dat7, "Mix - (20.0, 1.0, .01, .01)")
colnames(dat7) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat8 = data.frame($df24)
dat8 = cbind(dat8, "Mix - (10.0, .5, .5, .5)")
colnames(dat8) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat_mix <- rbind(dat1, dat2, dat3, dat4,dat6,dat7,dat8)
"""

R"""
dat1 = data.frame($df25)
dat1 = cbind(dat1, "PoisRE - N(5,5)")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat2 = data.frame($df26)
dat2 = cbind(dat2, "PoisRE - N(5,1)")
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat3 = data.frame($df27)
dat3 = cbind(dat3, "PoisRE - N(0,5)")
colnames(dat3) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat4 = data.frame($df28)
dat4 = cbind(dat4, "PoisRE - N(0,1)")
colnames(dat4) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat_re <- rbind(dat1, dat2, dat3, dat4)
"""

R"""
dat1 = data.frame($df29)
dat1 = cbind(dat1, "PoisRE - N(3,1)")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat2 = data.frame($df30)
dat2 = cbind(dat2, "PoisRE - N(3,2)")
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat3 = data.frame($df31)
dat3 = cbind(dat3, "PoisRE - N(3,5)")
colnames(dat3) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat4 = data.frame($df32)
dat4 = cbind(dat4, "PoisRE - N(3,10)")
colnames(dat4) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat_re1 <- rbind(dat1, dat2, dat3, dat4)
"""

R"""
dat1 = data.frame($df33)
dat1 = cbind(dat1, "NB2 - (1,1), (0.1,0.1)")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat2 = data.frame($df34)
dat2 = cbind(dat2, "NB2 - (1,1), (1,1)")
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat3 = data.frame($df35)
dat3 = cbind(dat3, "NB2 - (5,1), (0.1,0.1)")
colnames(dat3) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat4 = data.frame($df36)
dat4 = cbind(dat4, "NB2 - (5,1), (1,1)")
colnames(dat4) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat_nb21 <- rbind(dat1, dat2, dat3, dat4)
"""
R"""
dat1 = data.frame($df37)
dat1 = cbind(dat1, "NB2 - (20,1), (0.1,0.1)")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat2 = data.frame($df38)
dat2 = cbind(dat2, "NB2 - (10,1), (0.1,0.1)")
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat3 = data.frame($df39)
dat3 = cbind(dat3, "NB2 - (5,1), (0.1,0.1)")
colnames(dat3) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat4 = data.frame($df40)
dat4 = cbind(dat4, "NB2 - (1,1), (0.1,0.1)")
colnames(dat4) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat_nb22 <- rbind(dat1, dat2, dat3, dat4)
"""


R"""
p1 <- dat_1 %>%
    subset(Time < 3.1) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.5) + xlim(0,3) +
    geom_hline(yintercept = 0.2)
p2 <- dat_2 %>%
    subset(Time < 3.1) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7,2)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.5) + xlim(0,3) +
    geom_hline(yintercept = 0.2)
p3 <- dat_3 %>%
    subset(Time < 3.1) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.5) +
    geom_hline(yintercept = 0.2)
p4 <- dat_4 %>%
    subset(Time < 3.1) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.5) +
    geom_hline(yintercept = 0.2)

p5 <- dat_all %>%
    subset(Time < 3.1) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.5) + xlim(0,3) 

p6 <- dat_p %>%
    subset(Time < 3.1) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.5) + xlim(0,3) 

p7 <- dat_mix %>%
    subset(Time < 3.1) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7,1,2,3)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.5) + xlim(0,3) 

p8 <- dat_re %>%
    subset(Time < 3.1) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.5) + xlim(0,3) 
p9 <- dat_re1 %>%
    subset(Time < 3.1) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.5) + xlim(0,3) 

p10 <- dat_nb21 %>%
    subset(Time < 3.1) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.5) + xlim(0,3) 

p11 <- dat_nb22 %>%
    subset(Time < 3.1) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.5) + xlim(0,3) 

plot_grid(p1,p2,p3,p4)
plot_grid(p5,p6)
plot_grid(p8,p9)
plot_grid(p10,p11)
#ggsave($plotsdir("Priors_sen.pdf"), width = 8, height = 6)
"""

plot(out1["Sk_t"],out1["Sk_x"][1,10,:])

plot(out1["Sk_t"],out1["Sk_θ"][1,10,:])

plot(out1["Sk_θ"][1,10,:], log.(out1["Sk_σ"][1,:]))
ylabel!("σ");xlabel!("Centred θ")
plot(out1["Sk_x"][1,10,:], log.(out1["Sk_σ"][1,:]))
ylabel!("σ");xlabel!("Non-centred θ")

plot(out1["Sk_θ"][1,2,:], log.(out1["Sk_σ"][1,:]))
plot(out1["Sk_x"][1,2,:], log.(out1["Sk_σ"][1,:]))

plot(scatter(out1["Sk_θ"][1,1,:], out1["Sk_θ"][1,2,:]))
plot(scatter(out1["Sk_x"][1,1,:], out1["Sk_x"][1,2,:]))

plot(out4["Sk_t"],out4["Sk_Γ"])