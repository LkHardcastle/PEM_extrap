using DrWatson
@quickactivate "PEM_extrap"
# For src
using DataStructures, LinearAlgebra, Distributions, Random, Optim, Roots, SpecialFunctions, Statistics
using Plots, CSV, DataFrames, RCall, Interpolations, MCMCDiagnosticTools

include(srcdir("Sampler.jl"))
include(srcdir("PreProcessing.jl"))
include(srcdir("PostProcessing.jl"))

R"""
library(ggplot2)
library(dplyr)
library(tidyr)
library(cowplot)
library(coda)
cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
"""

Random.seed!(12515)
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
nits = 10_000
nsmp = 1_000
test_times = collect(0.05:0.05:2.95)
sampler = []
J_vec = []
h_mat = Matrix{Float64}(undef, 3, 0)
it = []
tuning = []
exp_its = 5
h_test = [0.5, 1.5, 2.5]
h_ess = Vector{Vector{Float64}}()


Random.seed!(12515)
priors1 = BasicPrior(1.0, FixedV(1.0), FixedW([0.5]), 1.0, CtsPois(10.0, 10.0, 100.0, 3.1), [RandomWalk()], [], 2)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
nits = 20_000
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 5.0, false, true, 0.01, 50.0)
@time out1 = pem_fit(state0, dat, priors1, settings, test_times, 1_000)

Random.seed!(12515)
nits = 1_000_000
priors3 = BasicPrior(1.0, FixedV(1.0), FixedW([0.5]), 0.0, RJ(10.0, 1, 0.01, 100.0, 3.1), [RandomWalk()], [], 2)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = RWM(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0), 0.05, 0)
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.05, 50.0)
@time out2 = pem_fit(state0, dat, priors3, settings, test_times, 10_000)

Random.seed!(12515)
nits = 1_000_000
priors3 = BasicPrior(1.0, FixedV(1.0), FixedW([0.5]), 0.0, RJ(5.0, 1, 1.0, 100.0, 3.1), [RandomWalk()], [], 2)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = RWM(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0), 0.05, 0)
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.05, 50.0)
@time out3 = pem_fit(state0, dat, priors3, settings, test_times, 10_000)

Random.seed!(12515)
nits = 1_000_000
priors3 = BasicPrior(1.0, FixedV(1.0), FixedW([0.5]), 0.0, RJ(5.0, 1, 0.25, 100.0, 3.1), [RandomWalk()], [], 2)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = RWM(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0), 0.05, 0)
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.05, 50.0)
@time out4 = pem_fit(state0, dat, priors3, settings, test_times, 10_000)

c1 = cts_transform(cumsum(out1[1]["Sk_θ"], dims = 2), out1[1]["Sk_s_loc"], h_test)
c2 = cts_transform(cumsum(out1[2]["Sk_θ"], dims = 2), out1[2]["Sk_s_loc"], h_test)
h1 = cts_transform(cumsum(out1[2]["Sk_θ"], dims = 2), out1[2]["Sk_s_loc"], collect(0.01:0.01:3.0))
h2 = cts_transform(cumsum(out1[1]["Sk_θ"], dims = 2), out1[1]["Sk_s_loc"], collect(0.01:0.01:3.0))
c1_ = cts_transform(cumsum(out2[1]["Sk_θ"], dims = 2), out2[1]["Sk_s_loc"], h_test)
c2_ = cts_transform(cumsum(out2[2]["Sk_θ"], dims = 2), out2[2]["Sk_s_loc"], h_test)
h1_ = cts_transform(cumsum(out2[1]["Sk_θ"][:,:,1:50:1_000_000], dims = 2), out2[1]["Sk_s_loc"][:,1:50:1_000_000], collect(0.01:0.01:3.0))
h2_ = cts_transform(cumsum(out2[2]["Sk_θ"][:,:,1:50:1_000_000], dims = 2), out2[2]["Sk_s_loc"][:,1:50:1_000_000], collect(0.01:0.01:3.0))
c11 = cts_transform(cumsum(out3[1]["Sk_θ"], dims = 2), out3[1]["Sk_s_loc"], h_test)
c21 = cts_transform(cumsum(out3[2]["Sk_θ"], dims = 2), out3[2]["Sk_s_loc"], h_test)
h11 = cts_transform(cumsum(out3[2]["Sk_θ"][:,:,1:50:1_000_000], dims = 2), out3[2]["Sk_s_loc"][:,1:50:1_000_000], collect(0.01:0.01:3.0))
h21 = cts_transform(cumsum(out3[1]["Sk_θ"][:,:,1:50:1_000_000], dims = 2), out3[1]["Sk_s_loc"][:,1:50:1_000_000], collect(0.01:0.01:3.0))
c12 = cts_transform(cumsum(out4[1]["Sk_θ"], dims = 2), out4[1]["Sk_s_loc"], h_test)
c22 = cts_transform(cumsum(out4[2]["Sk_θ"], dims = 2), out4[2]["Sk_s_loc"], h_test)
h12 = cts_transform(cumsum(out4[2]["Sk_θ"][:,:,1:50:1_000_000], dims = 2), out4[2]["Sk_s_loc"][:,1:50:1_000_000], collect(0.01:0.01:3.0))
h22 = cts_transform(cumsum(out4[1]["Sk_θ"][:,:,1:50:1_000_000], dims = 2), out4[1]["Sk_s_loc"][:,1:50:1_000_000], collect(0.01:0.01:3.0))

plot(collect(0.01:0.01:3.0), exp.(vec(median(h2[:,:,:], dims = 3))))
plot!(collect(0.01:0.01:3.0), exp.(quantile.(eachrow(h2[1,:,:]), 0.025)))
plot!(collect(0.01:0.01:3.0), exp.(quantile.(eachrow(h2[1,:,:]), 0.975)))
plot!(collect(0.01:0.01:3.0), exp.(vec(median(h2_[:,:,:], dims = 3))), linestyle = :dot)
plot!(collect(0.01:0.01:3.0), exp.(quantile.(eachrow(h2_[1,:,:]), 0.025)), linestyle = :dot)
plot!(collect(0.01:0.01:3.0), exp.(quantile.(eachrow(h2_[1,:,:]), 0.975)), linestyle = :dot)

plot(h1[1,120,1_000:end])
plot!(h1_[1,120,1_000:end])

df1 = DataFrame(PDMP = h1[1,120,1_000:end], RJ = h1_[1,120,1_000:end])
df_pdmp = DataFrame(Time = collect(0.01:0.01:3.0), median = exp.(vec(median(h2[:,:,:], dims = 3))), 
                    LCI = exp.(quantile.(eachrow(h2[1,:,:]), 0.025)), UCI = exp.(quantile.(eachrow(h2[1,:,:]), 0.975)), Method = "PDMP")
df_rj = DataFrame(Time = collect(0.01:0.01:3.0), median = exp.(vec(median(h1_[:,:,:], dims = 3))), 
                    LCI = exp.(quantile.(eachrow(h1_[1,:,:]), 0.025)), UCI = exp.(quantile.(eachrow(h1_[1,:,:]), 0.975)), Method = "RJ")
df2 = vcat(df_pdmp, df_rj)

CSV.write(datadir("RJexp", "trace_plots_main.csv"),df1)
CSV.write(datadir("RJexp", "hazards.csv"),df2)

df_rj_supp = DataFrame(Time = collect(0.01:0.01:3.0), median = exp.(vec(median(h11[:,:,:], dims = 3))), 
                    LCI = exp.(quantile.(eachrow(h11[1,:,:]), 0.025)), UCI = exp.(quantile.(eachrow(h11[1,:,:]), 0.975)), Method = "RJ")
CSV.write(datadir("RJexp", "hazards_supp.csv"),df_rj_supp)