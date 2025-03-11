using DrWatson
@quickactivate "PEM_extrap"
# For src
using DataStructures, LinearAlgebra, Distributions, Random, Optim, Roots, SpecialFunctions, Statistics
using Plots, CSV, DataFrames, Interpolations, MCMCDiagnosticTools, RCall

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
df = CSV.read(datadir("colon.csv"), DataFrame)
y = df.years
maximum(y)
n = length(y)
breaks = vcat(0.01,collect(0.26:0.25:3.01))
p = 1
cens = df.status
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 5_000
nsmp = 10
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 50.0)
test_times = collect(0.2:0.2:3.0)
λ = 1.0
γ = 2.0
μ = (t) -> (log(λ) .+ (γ-1).*log.(t))
μ = t -> log(0.5)
priors = BasicPrior(1.0, PC(1.0, 2, 0.5, Inf), FixedW([0.5]), 1.0, CtsPois(7.0, 1.0, 100.0, 3.1), [GaussLangevin(μ, 1.0)], [0.1], 2)
out1 = pem_fit(state0, dat, priors, settings, test_times)
println(out1[3]);println(out1[4])
breaks_extrap = collect(3.12:0.02:15)
grid = sort(unique(out1[1]["Sk_s_loc"][cumsum(out1[1]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
grid = vcat(0.0001, grid[1:5:length(grid)])
test_smp = cts_transform(cumsum(out1[1]["Sk_θ"], dims = 2), out1[1]["Sk_s_loc"], grid)
extrap1 = barker_extrapolation(out1[1], priors.diff[1], priors.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1, 0.1)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df1 = DataFrame(hcat(vcat(grid,breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)


λ = exp(-4.3)
γ = 2.1
μ = (t) -> (log(λ) .+ (γ-1).*log.(t))
priors = BasicPrior(1.0, PC(1.0, 2, 0.5, Inf), FixedW([0.5]), 1.0, CtsPois(7.0, 1.0, 100.0, 3.1), [GaussLangevin(μ, 1.0)], [0.1], 2)
out2 = pem_fit(state0, dat, priors, settings, test_times)
println(out2[3]);println(out2[4])
breaks_extrap = collect(3.12:0.02:15)
grid = sort(unique(out2[1]["Sk_s_loc"][cumsum(out2[1]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
grid = vcat(0.0001, grid[1:5:length(grid)])
test_smp = cts_transform(cumsum(out2[1]["Sk_θ"], dims = 2), out2[1]["Sk_s_loc"], grid)
extrap1 = barker_extrapolation(out2[1], priors.diff[1], priors.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1, 0.1)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df2 = DataFrame(hcat(vcat(grid,breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)


R"""
dat1 = data.frame($df1)
dat1 = cbind(dat1, "Gaussian")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat2 = data.frame($df2)
dat2 = cbind(dat2, "Weibull centred Gaussian")
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat_1 <- rbind(dat1, dat2)
"""


R"""
p1 <- dat_1 %>%
    subset(Time < 3.1) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.5) + xlim(0,3) 

p2 <- dat_1 %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,2.0) 
plot_grid(p1,p2, nrow = 1)
#ggsave($plotsdir("CovariateColon.pdf"), width = 8, height = 6)
"""
