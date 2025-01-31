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

# Look at σ sensitivity

Random.seed!(9102)
df = CSV.read(datadir("colon.csv"), DataFrame)
y = df.years
maximum(y)
n = length(y)
breaks = collect(0.01:0.1:3.11)
p = 1
cens = df.status
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 150000
nsmp = 10000
settings = Settings(nits, nsmp, 1_000_000, 1.0,0.5, 0.5, false, true)


priors1 = BasicPrior(1.0, PC([0.2], [0.1], [0.5], Inf), Beta([0.4], [10.0], [10.0]), 1.0, Cts(10.0, 50.0, 3.2), [RandomWalk()])
priors2 = BasicPrior(1.0, PC([0.2], [1.0], [0.5], Inf), Beta([0.4], [10.0], [10.0]), 1.0, Cts(10.0, 50.0, 3.2), [RandomWalk()])
priors3 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [10.0], [10.0]), 1.0, Cts(10.0, 50.0, 3.2), [RandomWalk()])
priors4 = BasicPrior(1.0, PC([0.2], [5.0], [0.5], Inf), Beta([0.4], [10.0], [10.0]), 1.0, Cts(10.0, 50.0, 3.2), [RandomWalk()])
Random.seed!(9102)
@time out1 = pem_sample(state0, dat, priors1, settings)
@time out2 = pem_sample(state0, dat, priors2, settings)
@time out3 = pem_sample(state0, dat, priors3, settings)
@time out4 = pem_sample(state0, dat, priors4, settings)

Random.seed!(1237)
grid = sort(unique(out1["Smp_s_loc"][cumsum(out1["Smp_s"],dims = 1)[1,:,:] .> 0.0]))
grid = grid[1:10:length(grid)]
breaks_extrap = collect(3.2:0.02:15)
extrap1 = barker_extrapolation(out1, priors1.diff[1], priors1.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out1["Smp_x"], dims = 2), out1["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df1 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out2, priors2.diff[1], priors2.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out2["Smp_x"], dims = 2), out2["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df2 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out3, priors3.diff[1], priors3.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out3["Smp_x"], dims = 2), out3["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df3 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out4, priors4.diff[1], priors4.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out4["Smp_x"], dims = 2), out4["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df4 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

CSV.write(datadir("ColonSmps","Sigma01.csv"), df1)
CSV.write(datadir("ColonSmps","Sigma1.csv"), df2)
CSV.write(datadir("ColonSmps","Sigma2.csv"), df3)
CSV.write(datadir("ColonSmps","Sigma5.csv"), df4)

# Look at ω sensitivity

Random.seed!(9102)
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
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 150_000
nsmp = 10000
settings = Settings(nits, nsmp, 1_000_000, 1.0,0.5, 0.5, false, true)


priors1 = BasicPrior(1.0, PC([0.05], [2], [0.5], Inf), Beta([0.4], [1.0], [9.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [RandomWalk()])
priors2 = BasicPrior(1.0, PC([0.05], [2], [0.5], Inf), Beta([0.4], [3.0], [7.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [RandomWalk()])
priors3 = BasicPrior(1.0, PC([0.05], [2], [0.5], Inf), Beta([0.4], [5.0], [5.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [RandomWalk()])
priors4 = BasicPrior(1.0, PC([0.05], [2], [0.5], Inf), Beta([0.4], [7.0], [3.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [RandomWalk()])
Random.seed!(9102)
@time out1 = pem_sample(state0, dat, priors1, settings)
@time out2 = pem_sample(state0, dat, priors2, settings)
@time out3 = pem_sample(state0, dat, priors3, settings)
@time out4 = pem_sample(state0, dat, priors4, settings)

x1, DIC1 = get_DIC(out1, dat)
x2, DIC2 = get_DIC(out2, dat)
x3, DIC3 = get_DIC(out3, dat)
x4, DIC4 = get_DIC(out4, dat)
plot(x1)
plot!(x2)
plot!(x3)
plot!(x4)

histogram(x1, fillalpha = 0.2)
histogram!(x2, fillalpha = 0.2)
histogram!(x3, fillalpha = 0.2)
histogram(x4, fillalpha = 0.2)

DIC1
mean(x1[findall(.!isnan.(x1))][2_000:end]) + 0.5*var(x1[findall(.!isnan.(x1))][2_000:end])
0.5*sqrt(var(x1[findall(.!isnan.(x1))][2_000:end]))
mean(x1[findall(.!isnan.(x1))][2_000:end]) + 0.5*sqrt(var(x1[findall(.!isnan.(x1))][2_000:end]))
DIC2
mean(x2[findall(.!isnan.(x2))][2_000:end]) + 0.5*var(x2[findall(.!isnan.(x2))][2_000:end])
0.5*sqrt(var(x2[findall(.!isnan.(x2))][2_000:end]))
mean(x2[findall(.!isnan.(x2))][2_000:end]) + 0.5*sqrt(var(x2[findall(.!isnan.(x2))][2_000:end]))
DIC3
mean(x3[findall(.!isnan.(x3))][2_000:end]) + 0.5*var(x3[findall(.!isnan.(x3))][2_000:end])
0.5*sqrt(var(x3[findall(.!isnan.(x3))][2_000:end]))
mean(x3[findall(.!isnan.(x3))][2_000:end]) + 0.5*sqrt(var(x3[findall(.!isnan.(x3))][2_000:end]))
DIC4
mean(x4[findall(.!isnan.(x4))][2_000:end]) + 0.5*var(x4[findall(.!isnan.(x4))][2_000:end])
0.5*sqrt(var(x4[findall(.!isnan.(x4))][2_000:end]))
mean(x4[findall(.!isnan.(x4))][2_000:end]) + 0.5*sqrt(var(x4[findall(.!isnan.(x4))][2_000:end]))



histogram(out1["Smp_σ"][1,:])
histogram!(out2["Smp_σ"][1,:], fillalpha = 0.5)
histogram!(out3["Smp_σ"][1,:], fillalpha = 0.5)
histogram!(out4["Smp_σ"][1,:], fillalpha = 0.5)

plot(log.(out1["Smp_σ"][1,:]))
plot!(out2["Smp_σ"][1,:])
plot!(out3["Smp_σ"][1,:])
plot!(out4["Smp_σ"][1,:])

plot(out1["Smp_ω"][1,:])
plot!(out2["Smp_ω"][1,:])
plot!(out3["Smp_ω"][1,:])
plot!(out4["Smp_ω"][1,:])

plot(test_smp[1,1200,:])

Random.seed!(1237)
grid = sort(unique(out1["Smp_s_loc"][cumsum(out1["Smp_s"],dims = 1)[1,:,:] .> 0.0]))
grid = grid[1:10:length(grid)]
breaks_extrap = collect(3.2:0.02:15)
extrap1 = barker_extrapolation(out1, priors1.diff[1], priors1.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out1["Smp_x"], dims = 2), out1["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df5 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

m1 = get_meansurv(out1["Smp_x"], out1["Smp_s_loc"], out1["Smp_J"], [1])
m2 = get_meansurv(reshape(extrap1,1,size(extrap1,1),size(extrap1,2)), stack(fill(breaks_extrap,size(extrap1,2))), fill(size(breaks_extrap,1),size(extrap1,2)), [1])
quantile(m1 .+ m2, 0.025)
quantile(m1 .+ m2, 0.975)
median(m1)
median(m2)
median(m1 .+ m2)

extrap1 = barker_extrapolation(out2, priors2.diff[1], priors2.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out2["Smp_x"], dims = 2), out2["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df6 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

m1 = get_meansurv(out2["Smp_x"], out2["Smp_s_loc"], out2["Smp_J"], [1])
m2 = get_meansurv(reshape(extrap1,1,size(extrap1,1),size(extrap1,2)), stack(fill(breaks_extrap,size(extrap1,2))), fill(size(breaks_extrap,1),size(extrap1,2)), [1])
quantile(m1 .+ m2, 0.025)
quantile(m1 .+ m2, 0.975)
median(m1)
median(m2)
median(m1 .+ m2)

extrap1 = barker_extrapolation(out3, priors3.diff[1], priors3.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out3["Smp_x"], dims = 2), out3["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df7 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

m1 = get_meansurv(out3["Smp_x"], out3["Smp_s_loc"], out3["Smp_J"], [1])
m2 = get_meansurv(reshape(extrap1,1,size(extrap1,1),size(extrap1,2)), stack(fill(breaks_extrap,size(extrap1,2))), fill(size(breaks_extrap,1),size(extrap1,2)), [1])
quantile(m1 .+ m2, 0.025)
quantile(m1 .+ m2, 0.975)
median(m1)
median(m2)
median(m1 .+ m2)

extrap1 = barker_extrapolation(out4, priors4.diff[1], priors4.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out4["Smp_x"], dims = 2), out4["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df8 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

m1 = get_meansurv(out4["Smp_x"], out4["Smp_s_loc"], out4["Smp_J"], [1])
m2 = get_meansurv(reshape(extrap1,1,size(extrap1,1),size(extrap1,2)), stack(fill(breaks_extrap,size(extrap1,2))), fill(size(breaks_extrap,1),size(extrap1,2)), [1])
quantile(m1 .+ m2, 0.025)
quantile(m1 .+ m2, 0.975)
median(m1)
median(m2)
median(m1 .+ m2)

CSV.write(datadir("ColonSmps","Beta1_1.csv"), df5)
CSV.write(datadir("ColonSmps","Beta5_5.csv"), df6)
CSV.write(datadir("ColonSmps","Beta1_10.csv"), df7)
CSV.write(datadir("ColonSmps","Beta10_1.csv"), df8)

extrap1

# Look at ω sensitivity

Random.seed!(9102)
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
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 150_000
nsmp = 10000
settings = Settings(nits, nsmp, 1_000_000, 1.0,0.5, 0.5, false, true)


priors1 = BasicPrior(1.0, PC([0.05], [2], [0.5], Inf), Beta([0.4], [1.0], [1.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [RandomWalk()])
priors2 = BasicPrior(1.0, PC([0.05], [2], [0.5], Inf), Beta([0.4], [0.5], [0.5]), 1.0, CtsPois(10.0, 50.0, 3.2), [RandomWalk()])
priors3 = BasicPrior(1.0, PC([0.05], [2], [0.5], Inf), Beta([0.4], [1/3], [1/3]), 1.0, CtsPois(10.0, 50.0, 3.2), [RandomWalk()])
priors4 = BasicPrior(1.0, PC([0.05], [2], [0.5], Inf), Beta([0.4], [0.1], [0.1]), 1.0, CtsPois(10.0, 50.0, 3.2), [RandomWalk()])
Random.seed!(9102)
@time out1 = pem_sample(state0, dat, priors1, settings)
@time out2 = pem_sample(state0, dat, priors2, settings)
@time out3 = pem_sample(state0, dat, priors3, settings)
@time out4 = pem_sample(state0, dat, priors4, settings)

Random.seed!(1237)
grid = sort(unique(out1["Smp_s_loc"][cumsum(out1["Smp_s"],dims = 1)[1,:,:] .> 0.0]))
grid = grid[1:10:length(grid)]
breaks_extrap = collect(3.2:0.02:15)
extrap1 = barker_extrapolation(out1, priors1.diff[1], priors1.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out1["Smp_x"], dims = 2), out1["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df9 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out2, priors2.diff[1], priors2.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out2["Smp_x"], dims = 2), out2["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df10 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out3, priors3.diff[1], priors3.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out3["Smp_x"], dims = 2), out3["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df11 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out4, priors4.diff[1], priors4.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out4["Smp_x"], dims = 2), out4["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df12 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

CSV.write(datadir("ColonSmps","BetaUn1_1.csv"), df9)
CSV.write(datadir("ColonSmps","BetaUn2_2.csv"), df10)
CSV.write(datadir("ColonSmps","BetaUn3_3.csv"), df11)
CSV.write(datadir("ColonSmps","BetaUn_10_10.csv"), df12)

Random.seed!(9102)
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
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 50000
nsmp = 10000
settings = Settings(nits, nsmp, 1_000_000, 1.0,0.5, 0.5, false, true)
seq = 0.1:0.1:10
plot(seq, pdf.(Gamma(5,1),seq))


priors1 = BasicPrior(1.0, PC([0.05], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB(5.0, 1.0, 10.0, 150.0, 3.2), [RandomWalk()])
priors2 = BasicPrior(1.0, PC([0.05], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB(10.0, 1.0, 10.0, 150.0, 3.2), [RandomWalk()])
priors3 = BasicPrior(1.0, PC([0.05], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB(10.0, 0.5, 10.0, 150.0, 3.2), [RandomWalk()])
priors4 = BasicPrior(1.0, PC([0.05], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsNB(10.0, 2.0, 10.0, 150.0, 3.2), [RandomWalk()])
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
test_smp = cts_transform(cumsum(out1["Smp_x"], dims = 2), out1["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df9 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out2, priors2.diff[1], priors2.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out2["Smp_x"], dims = 2), out2["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df10 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out3, priors3.diff[1], priors3.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out3["Smp_x"], dims = 2), out3["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df11 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out4, priors4.diff[1], priors4.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out4["Smp_x"], dims = 2), out4["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df12 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

CSV.write(datadir("ColonSmps","BetaNB1_1.csv"), df9)
CSV.write(datadir("ColonSmps","BetaNB5_5.csv"), df10)
CSV.write(datadir("ColonSmps","BetaNB1_10.csv"), df11)
CSV.write(datadir("ColonSmps","BetaNB10_1.csv"), df12)


df1 = CSV.read(datadir("ColonSmps","Sigma01.csv"), DataFrame)
df2 = CSV.read(datadir("ColonSmps","Sigma1.csv"), DataFrame)
df3 = CSV.read(datadir("ColonSmps","Sigma2.csv"), DataFrame)
df4 = CSV.read(datadir("ColonSmps","Sigma5.csv"), DataFrame)
df5 = CSV.read(datadir("ColonSmps","Beta1_1.csv"), DataFrame)
df6 = CSV.read(datadir("ColonSmps","Beta5_5.csv"), DataFrame)
df7 = CSV.read(datadir("ColonSmps","Beta1_10.csv"), DataFrame)
df8 = CSV.read(datadir("ColonSmps","Beta10_1.csv"), DataFrame)
df9 = CSV.read(datadir("ColonSmps","BetaUn1_1.csv"), DataFrame)
df10 = CSV.read(datadir("ColonSmps","BetaUn2_2.csv"), DataFrame)
df11 = CSV.read(datadir("ColonSmps","BetaUn3_3.csv"), DataFrame)
df12 = CSV.read(datadir("ColonSmps","BetaUn_10_10.csv"), DataFrame)
df9 = CSV.read(datadir("ColonSmps","BetaNB1_1.csv"), DataFrame)
df10 = CSV.read(datadir("ColonSmps","BetaNB5_5.csv"), DataFrame)
df11 = CSV.read(datadir("ColonSmps","BetaNB1_10.csv"), DataFrame)
df12 = CSV.read(datadir("ColonSmps","BetaNB10_1.csv"), DataFrame)


R"""
dat1 = data.frame($df9)
dat1 = cbind(dat1, "(5,1)")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat2 = data.frame($df10)
dat2 = cbind(dat2, "(10,1)")
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat3 = data.frame($df11)
dat3 = cbind(dat3, "(5,1/2)")
colnames(dat3) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat4 = data.frame($df12)
dat4 = cbind(dat4, "(10,2)")
colnames(dat4) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat_sigma <- rbind(dat1, dat2, dat3, dat4)
"""

R"""
dat5 = data.frame($df5)
dat5 = cbind(dat5, "(1,9)")
colnames(dat5) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat6 = data.frame($df6)
dat6 = cbind(dat6, "(3,7)")
colnames(dat6) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat7 = data.frame($df7)
dat7 = cbind(dat7, "(5,5)")
colnames(dat7) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat8 = data.frame($df8)
dat8 = cbind(dat8, "(7,3)")
colnames(dat8) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat9 = data.frame($df11)
dat9 = cbind(dat9, "(1/3,1/3)")
colnames(dat9) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")
dat_beta <- rbind(dat5, dat6, dat7, dat8, dat9)
"""

R"""
p1 <- dat_sigma %>%
    subset(Time < 3.1) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.5) + xlim(0,3)
p2 <- dat_beta %>%
    subset(Time < 3.1) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7,2)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.5) + xlim(0,3)
p3 <- dat_sigma %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.5)
p4 <- dat_beta %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7,2)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.5)
plot_grid(p1,p2,p3,p4, nrow = 2)
plot_grid(p1,p3)
#plot_grid(p1,p3)
#plot_grid(p1,p2)
#ggsave($plotsdir("Priors_sen.pdf"), width = 8, height = 6)
"""

histogram(out1["Smp_J"])
histogram(out2["Smp_J"])
histogram(out3["Smp_J"])
histogram(out4["Smp_J"])

out1["Smp_s_loc"]