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
nits = 150000
nsmp = 20000
settings = Settings(nits, nsmp, 1_000_000, 1.0,0.5, 0.5, false, true)


priors1 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [10.0], [10.0]), 1.0, Cts(10.0, 50.0, 3.5), [RandomWalk()])
priors2 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [10.0], [10.0]), 1.0, Cts(10.0, 50.0, 3.5), [GaussLangevin(-1.0,1.0)])
priors3 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [10.0], [10.0]), 1.0, Cts(10.0, 50.0, 3.5), [GammaLangevin(0.5,2)])
priors4 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [10.0], [10.0]), 1.0, Cts(10.0, 50.0, 3.5), [GompertzBaseline(0.5)])
Random.seed!(9102)
@time out1 = pem_sample(state0, dat, priors1, settings)
@time out2 = pem_sample(state0, dat, priors2, settings)
@time out3 = pem_sample(state0, dat, priors3, settings)
@time out4 = pem_sample(state0, dat, priors4, settings)

grid = collect(0.01:0.01:3.2)
test_smp = cts_transform(cumsum(out1["Smp_x"], dims = 2), out1["Smp_s_loc"], grid)
s1 = view(exp.(test_smp), 1, :, :)
df1 = DataFrame(hcat(grid, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
test_smp = cts_transform(cumsum(out2["Smp_x"], dims = 2), out2["Smp_s_loc"], grid)
s1 = view(exp.(test_smp), 1, :, :)
df2 = DataFrame(hcat(grid, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
test_smp = cts_transform(cumsum(out3["Smp_x"], dims = 2), out3["Smp_s_loc"], grid)
s1 = view(exp.(test_smp), 1, :, :)
df3 = DataFrame(hcat(grid, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
test_smp = cts_transform(cumsum(out4["Smp_x"], dims = 2), out4["Smp_s_loc"], grid)
s1 = view(exp.(test_smp), 1, :, :)
df4 = DataFrame(hcat(grid, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

Random.seed!(1237)
breaks_extrap = collect(3.2:0.02:15)
extrap1 = barker_extrapolation(out1, priors1.diff[1], priors1.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out1["Smp_x"], dims = 2), out1["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df5 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out2, priors2.diff[1], priors2.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out2["Smp_x"], dims = 2), out2["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df6 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out3, priors3.diff[1], priors3.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out3["Smp_x"], dims = 2), out3["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df7 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out4, priors4.diff[1], priors4.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
test_smp = cts_transform(cumsum(out4["Smp_x"], dims = 2), out4["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
df8 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

CSV.write(datadir("ColonSmps","RW.csv"), df5)
CSV.write(datadir("ColonSmps","Gaussian.csv"), df6)
CSV.write(datadir("ColonSmps","Gamma.csv"), df7)
CSV.write(datadir("ColonSmps","Gompertz.csv"), df8)

R"""
dat1 = data.frame($df1)
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI") 
dat2 = data.frame($df2)
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI") 
dat3 = data.frame($df3)
colnames(dat3) <- c("Time","Mean","LCI","Q1","Q4","UCI") 
dat4 = data.frame($df4)
colnames(dat4) <- c("Time","Mean","LCI","Q1","Q4","UCI") 
dat5 = data.frame($df5)
colnames(dat5) <- c("Time","Mean","LCI","Q1","Q4","UCI") 
dat6 = data.frame($df6)
colnames(dat6) <- c("Time","Mean","LCI","Q1","Q4","UCI") 
"""

R"""    
p1 <- dat1 %>%
    pivot_longer(Mean:UCI) %>%
    ggplot(aes(x = Time, y = value, col = name,linetype = name)) + geom_step() +
    theme_classic() +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,6,6,6)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,NA)
p2 <- dat2 %>%
    pivot_longer(Mean:UCI) %>%
    ggplot(aes(x = Time, y = value, col = name,linetype = name)) + geom_step() +
    theme_classic() +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,6,6,6)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,NA)
p3 <- dat3 %>%
    pivot_longer(Mean:UCI) %>%
    ggplot(aes(x = Time, y = value, col = name,linetype = name)) + geom_step() +
    theme_classic() +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,6,6,6)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,NA)
p4 <- dat4 %>%
    pivot_longer(Mean:UCI) %>%
    ggplot(aes(x = Time, y = value, col = name,linetype = name)) + geom_step() +
    theme_classic() +
    geom_vline(aes(xintercept = 3), linetype = "dotted") +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,6,6,6)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,NA)
p5 <- dat5 %>%
    pivot_longer(Mean:UCI) %>%
    ggplot(aes(x = Time, y = value, col = name,linetype = name)) + geom_step() +
    theme_classic() +
    geom_vline(aes(xintercept = 3), linetype = "dotted") +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,6,6,6)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,NA)
p6 <- dat6 %>%
    pivot_longer(Mean:UCI) %>%
    ggplot(aes(x = Time, y = value, col = name,linetype = name)) + geom_step() +
    theme_classic() +
    geom_vline(aes(xintercept = 3), linetype = "dotted") +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,6,6,6)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,NA)
plot_grid(p1,p2,p3,p4,p5,p6, nrow = 2)
#ggsave($plotsdir("Colon","Colon5.pdf"), width = 8, height = 6)
"""