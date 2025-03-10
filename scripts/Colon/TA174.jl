using DrWatson
@quickactivate "PEM_extrap"
# For src
using DataStructures, LinearAlgebra, Distributions, Random, Optim, Roots, SpecialFunctions, Statistics
using Plots, CSV, DataFrames, RCall, Interpolations, MCMCDiagnosticTools, ParetoSmooth

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

Random.seed!(2352)
df = CSV.read(datadir("TA174.csv"), DataFrame)
y = df.death_ty
maximum(y)
n = length(y)
breaks = vcat(0.1,collect(0.26:0.25:4.01))
p = 1
cens = df.death
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 10_000
nsmp = 10
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
priors = BasicPrior(1.0, PC([1.0], [2], [0.5], Inf), FixedW([0.5]), 1.0, CtsPois(5.0, 1.0, 100.0, 4.1), [RandomWalk()], [0.1], 2)
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 50.0)
out = pem_fit(state0, dat, priors, settings, test_times)

grid = sort(unique(out[1]["Sk_s_loc"][cumsum(out[1]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
grid = grid[1:10:length(grid)]
test_smp = cts_transform(cumsum(out[1]["Sk_θ"], dims = 2), out[1]["Sk_s_loc"], grid)
s1 = view(exp.(test_smp), 1, :, :)
df11 = DataFrame(hcat(grid, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

R"""
dat1 = data.frame($df11)
dat1 = cbind(dat1, "Pois(5)")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
p1 <- dat1 %>%
    subset(Time < 4.1) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.3) + xlim(0,4) 
"""


Random.seed!(2352)
df = CSV.read(datadir("TA174.csv"), DataFrame)
y = df.death_ty
maximum(y)
n = length(y)
breaks = vcat(0.1,collect(0.26:1.0:4.26))
p = 1
cens = df.death
covar = fill(1.0, 1, n)
trt = (df.treat .== 1)
covar = [covar; transpose(trt)]
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
x0[2,:] = vcat(x0[2,1], zeros(size(breaks) .-1))
v0[2,:] = vcat(v0[2,1], 1.0, zeros(size(breaks) .-2))
s0[2,:] = vcat(s0[2,1], true, zeros(Int,size(breaks) .-2))
v0 = v0./norm(v0)
nits = 5_000
nsmp = 10
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
#priors = BasicPrior(1.0, FixedV([0.5,0.5]), FixedW([0.5,0.5]), 1.0, CtsPois(5.0, 1.0, 100.0, 4.1), [RandomWalk(), RandomWalk()], [], 2)
priors = BasicPrior(1.0, PC([1.0,1.0], [2,2], [0.5,0.5], Inf), FixedW([0.5,0.5]), 1.0, CtsPois(5.0, 1.0, 100.0, 4.1), [RandomWalk(), RandomWalk()], [0.01,0.01], 2)
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 50.0)
out = pem_fit(state0, dat, priors, settings, test_times)
println(out[3]);println(out[4])

plot(out[1]["Sk_θ"][1,5,:])

grid = sort(unique(out[1]["Sk_s_loc"][cumsum(out[1]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
grid = grid[1:1:length(grid)]
test_smp = cts_transform(cumsum(out[1]["Sk_θ"], dims = 2), out[1]["Sk_s_loc"], grid)
s1 = view(exp.(test_smp), 1, :, :)
df1 = DataFrame(hcat(grid, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

s2 = exp.(view(test_smp, 1, :, :) .+ view(test_smp, 2, :, :))
df2 = DataFrame(hcat(grid, median(s2, dims = 2), quantile.(eachrow(s2), 0.025), quantile.(eachrow(s2), 0.25), quantile.(eachrow(s2), 0.75), quantile.(eachrow(s2), 0.975)), :auto)

R"""
dat1 = data.frame($df1)
dat1 = cbind(dat1, "Pois(5)")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
p1 <- dat1 %>%
    subset(Time < 4.1) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.3) + xlim(0,4) 
dat1 = data.frame($df2)
dat1 = cbind(dat1, "Pois(5)")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
p2 <- dat1 %>%
    subset(Time < 4.1) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.3) + xlim(0,4) 
plot_grid(p1,p2)
"""