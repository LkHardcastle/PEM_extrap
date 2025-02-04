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

Pois = [1.0, 5.0, 10.0, 15.0, 20.0]
Diff = [RandomWalk(), GaussLangevin(log(0.2),1.0), GammaLangevin(2,7), GompertzBaseline(0.5)]

Random.seed!(24562)
i = 1
j = 1
DIC_vec = []
MeanSurv = DataFrame(diff = [], pois = [], mean_obs = [], q1_obs = [], q2_obs = [], mean_ext = [], q1_ext = [], q2_ext = [], DIC = [])
for diff in eachindex(Diff)
    for fish in eachindex(Pois)
        priors = BasicPrior(1.0, PC([0.05], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPois(fish, 150.0, 3.2), [diff])
        out1 = pem_sample(state0, dat, priors, settings)
        grid = collect(0.02:0.02:3.05)
        breaks_extrap = collect(3.05:0.02:15)
        extrap1 = barker_extrapolation(out1, priors1.diff[1], priors1.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)[:,floor(Int,0.5*nsmp):end]
        test_smp = cts_transform(cumsum(out1["Smp_x"], dims = 2), out1["Smp_s_loc"], grid)[:,:,floor(Int,0.5*nsmp):end]
        s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
        df1 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
        CSV.write(datadir("ColonSmps","Pois$i$j.csv"), df1)
        m1 = get_meansurv(out1["Smp_x"], out1["Smp_s_loc"], out1["Smp_J"], [1])
        m2 = get_meansurv(reshape(extrap1,1,size(extrap1,1),size(extrap1,2)), stack(fill(breaks_extrap,size(extrap1,2))), fill(size(breaks_extrap,1),size(extrap1,2)), [1])
        push!(MeanSurv, [i j mean(m1) quantile(m1, 0.025) quantile(m1, 0.0975) mean(m2) quantile(m2, 0.025) quantile(m2, 0.0975) get_DIC(out1, dat)[2]])
        j += 1
    end
    i += 1
    j = 1
end
CSV.write(datadir("ColonSmps","PoisMeanSurv.csv"), mean_surv_obs)
