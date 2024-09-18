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


Random.seed!(9102)
df_all = CSV.read(datadir("SOLVD","SOLVD.csv"), DataFrame)
df = filter(idx -> idx.CLINIC == 1, df_all)
df.FUTIME/365
sum(df.EPX)
y = df.FUTIME/365
maximum(y)
n = length(y)
breaks = collect(0.1:0.1:5.1)
p = 1
cens = df.EPX
sum(cens)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 150000
nsmp = 20000
settings = Settings(nits, nsmp, 1_000_000, 1.0,0.5, 0.5, false, true)

histogram(y)

priors1 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [10.0], [10.0]), 1.0, Cts(5.0, 150.0, 5.2), [RandomWalk()])
priors2 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [10.0], [10.0]), 1.0, Cts(5.0, 150.0, 5.2), [GaussLangevin(-1.0,1.0)])
priors3 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [10.0], [10.0]), 1.0, Cts(5.0, 150.0, 5.2), [GammaLangevin(0.5,2)])
priors4 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [10.0], [10.0]), 1.0, Cts(5.0, 150.0, 5.2), [GompertzBaseline(1.0)])
Random.seed!(9102)
@time out1 = pem_sample(state0, dat, priors1, settings)
@time out2 = pem_sample(state0, dat, priors2, settings)
@time out3 = pem_sample(state0, dat, priors3, settings)
@time out4 = pem_sample(state0, dat, priors4, settings)

Random.seed!(1237)
grid = collect(0.02:0.02:5.198)
breaks_extrap = collect(5.2:0.02:15)
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

histogram(out1["Smp_s_loc"])
R"""
dat1 = data.frame($df1)
dat1 = cbind(dat1, "Random Walk")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat2 = data.frame($df2)
dat2 = cbind(dat2, "Log-Normal Langevin")
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat3 = data.frame($df3)
dat3 = cbind(dat3, "Gamma Langevin")
colnames(dat3) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat4 = data.frame($df4)
dat4 = cbind(dat4, "Gompertz dynamics")
colnames(dat4) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat_diffusion <- rbind(dat1, dat2, dat3, dat4)
"""

R"""
p1 <- dat_diffusion %>%
    subset(Time < 5.2) %>%
    #subset(Model == "Random Walk")  %>%
    pivot_longer(Mean:UCI,) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,1) + xlim(0,5.2)
p2 <- dat_diffusion %>%
    #subset(Time < 5.2) %>%
    #subset(Model == "Random Walk")  %>%
    pivot_longer(Mean:UCI,) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,5) + xlim(0,15) +
    geom_vline(aes(xintercept = 5.2), linetype = "dotted")
"""