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
Random.seed!(123)
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
state0 = ECMC2(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks), true, findall(s0), ones(size(x0)))
nits = 100000
nsmp = 100000
settings = Settings(nits, nsmp, 1_000_000, 0.5,0.5, 0.5, false, true)

priors1 = BasicPrior(1.0, FixedV(0.5), FixedW(0.5), 0.0, Fixed(0.1), [RandomWalk()])
priors2 = BasicPrior(1.0, FixedV(0.5), FixedW(0.5), 0.0, Fixed(0.1), [GaussLangevin(-1.0,1.0)])
priors3 = BasicPrior(1.0, FixedV(0.5), FixedW(0.5), 0.0, Fixed(0.1), [GammaLangevin(0.5,2)])
Random.seed!(9102)
@time out1 = pem_sample(state0, dat, priors1, settings)
Random.seed!(9102)
@time out2 = pem_sample(state0, dat, priors2, settings)
Random.seed!(9102)
@time out3 = pem_sample(state0, dat, priors3, settings)

priors4 = BasicPrior(1.0, FixedV(0.5), FixedW(0.5), 1.0, Fixed(0.1), [RandomWalk()])
priors5 = BasicPrior(1.0, FixedV(0.5), FixedW(0.5), 1.0, Fixed(0.1), [GaussLangevin(-1.0,1.0)])
priors6 = BasicPrior(1.0, FixedV(0.5), FixedW(0.5), 1.0, Fixed(0.1), [GammaLangevin(0.5,2)])
Random.seed!(9102)
@time out4 = pem_sample(state0, dat, priors4, settings)
@time out5 = pem_sample(state0, dat, priors5, settings)
@time out6 = pem_sample(state0, dat, priors6, settings)

priors7 = BasicPrior(1.0, PC(0.2, 2, 0.5, 1, Inf), Beta(0.4, 10.0, 10.0), 1.0, Fixed(0.1), [RandomWalk()])
priors8 = BasicPrior(1.0, PC(0.2, 2, 0.5, 1, Inf), Beta(0.4, 10.0, 10.0), 1.0, Fixed(0.1), [GaussLangevin(-1.0,1.0)])
priors9 = BasicPrior(1.0, PC(0.2, 2, 0.5, 1, Inf), Beta(0.4, 10.0, 10.0), 1.0, Fixed(0.1), [GammaLangevin(0.5,2)])
Random.seed!(9102)
@time out7 = pem_sample(state0, dat, priors7, settings)
@time out8 = pem_sample(state0, dat, priors8, settings)
@time out9 = pem_sample(state0, dat, priors9, settings)

priors10 = BasicPrior(1.0, FixedV(0.5), FixedW(0.5), 1.0, Cts(10.0, 50.0, 3.5), [RandomWalk()])
priors11 = BasicPrior(1.0, FixedV(0.5), FixedW(0.5), 1.0, Cts(10.0, 50.0, 3.5), [GaussLangevin(-1.0,1.0)])
priors12 = BasicPrior(1.0, FixedV(0.5), FixedW(0.5), 1.0, Cts(10.0, 50.0, 3.5), [GammaLangevin(0.5,2)])
Random.seed!(9102)
@time out10 = pem_sample(state0, dat, priors10, settings)
@time out11 = pem_sample(state0, dat, priors11, settings)
@time out12 = pem_sample(state0, dat, priors12, settings)

priors13 = BasicPrior(1.0, PC(0.2, 2, 0.5, 1, Inf), Beta(0.4, 10.0, 10.0), 1.0, Cts(10.0, 50.0, 3.5), [RandomWalk()])
priors14 = BasicPrior(1.0, PC(0.2, 2, 0.5, 1, Inf), Beta(0.4, 10.0, 10.0), 1.0, Cts(10.0, 50.0, 3.5),[GaussLangevin(-2.0,0.5)])
priors15 = BasicPrior(1.0, PC(0.2, 2, 0.5, 1, Inf), Beta(0.4, 10.0, 10.0), 1.0, Cts(10.0, 50.0, 3.5), [GammaLangevin(2,4)])
Random.seed!(9102)
@time out13 = pem_sample(state0, dat, priors13, settings)
@time out14 = pem_sample(state0, dat, priors14, settings)
@time out15 = pem_sample(state0, dat, priors15, settings)

df = CSV.read(datadir("colon.csv"), DataFrame)
y = df.years
maximum(y)
n = length(y)
breaks = collect(0.1:0.1:3.1)
p = 1
cens = df.status
covar = fill(1.0, 1, n)
covar = [covar; transpose(df.obstruct)]
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
x0[2,:] = vcat(x0[2,1],zeros(size(breaks) .-1))
v0[2,:] = vcat(v0[2,1],zeros(size(breaks) .-1))
s0[2,:] = vcat(s0[2,1],zeros(Int,size(breaks) .-1))
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, .!s0, breaks, t0, length(breaks), true, findall(s0), ones(size(x0)))
nits = 150000
nsmp = 20000
settings = Settings(nits, nsmp, 1_000_000, 1.0,0.5, 0.5, false, true)
Random.seed!(9102)
priors16 = BasicPrior(1.0, PC([0.2,0.2], [2,2], [0.5,0.5], Inf), Beta([0.3,0.3], [10.0,10.0], [10.0,10.0]), 1.0, Cts(5.0, 100.0, 3.2), [RandomWalk(), RandomWalk()])
@time out16 = pem_sample(state0, dat, priors16, settings)
priors17 = BasicPrior(1.0, PC([0.2,0.2], [2,2], [0.5,0.5], Inf), Beta([0.3,0.3], [10.0,10.0], [10.0,10.0]), 1.0, Cts(5.0, 100.0, 3.2), [GammaLangevin(2.0,4.0), GaussLangevin(0.0,0.5)])
Random.seed!(9102)
@time out17 = pem_sample(state0, dat, priors17, settings)
priors18 = BasicPrior(1.0, PC([0.2,0.2], [2,2], [0.5,0.5], Inf), Beta([0.3,0.3], [10.0,10.0], [10.0,10.0]), 1.0, Cts(5.0, 100.0, 3.2), [GammaLangevin(2.0,4.0), GaussLangevin(1.0,0.5)])
Random.seed!(9102)
@time out18 = pem_sample(state0, dat, priors18, settings)

s1 = view(exp.(out1["Smp_trans"]), 1, :, :)
df1 = DataFrame(hcat(breaks, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

s1 = view(exp.(out2["Smp_trans"]), 1, :, :)
df2 = DataFrame(hcat(breaks, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

s1 = view(exp.(out3["Smp_trans"]), 1, :, :)
df3 = DataFrame(hcat(breaks, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

Random.seed!(1237)
breaks_extrap = collect(3.2:0.1:20)
extrap1 = barker_extrapolation(out1, priors1.diff[1], priors1.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap)
s1 = vcat(view(exp.(out1["Smp_trans"]), 1, :, :), view(exp.(extrap1), 1, :, :))
df4 = DataFrame(hcat(vcat(breaks, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out2, priors2.diff[1], priors2.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap)
s1 = vcat(view(exp.(out2["Smp_trans"]), 1, :, :), view(exp.(extrap1), 1, :, :))
df5 = DataFrame(hcat(vcat(breaks, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out3, priors3.diff[1], priors3.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap)
s1 = vcat(view(exp.(out3["Smp_trans"]), 1, :, :), view(exp.(extrap1), 1, :, :))
df6 = DataFrame(hcat(vcat(breaks, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
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
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,6,6,6)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,NA)
p5 <- dat5 %>%
    pivot_longer(Mean:UCI) %>%
    ggplot(aes(x = Time, y = value, col = name,linetype = name)) + geom_step() +
    theme_classic() +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,6,6,6)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,NA)
p6 <- dat6 %>%
    pivot_longer(Mean:UCI) %>%
    ggplot(aes(x = Time, y = value, col = name,linetype = name)) + geom_step() +
    theme_classic() +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,6,6,6)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,NA)
plot_grid(p1,p2,p3,p4,p5,p6, nrow = 2)
#ggsave($plotsdir("Colon","Colon1.pdf"), width = 8, height = 6)
"""

s1 = view(exp.(out4["Smp_trans"]), 1, :, :)
df1 = DataFrame(hcat(breaks, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
s1 = view(exp.(out5["Smp_trans"]), 1, :, :)
df2 = DataFrame(hcat(breaks, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
s1 = view(exp.(out6["Smp_trans"]), 1, :, :)
df3 = DataFrame(hcat(breaks, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

Random.seed!(1237)
breaks_extrap = collect(3.2:0.1:20)
extrap1 = barker_extrapolation(out4, priors4.diff, priors4.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap)
s1 = vcat(view(exp.(out4["Smp_trans"]), 1, :, :), view(exp.(extrap1), 1, :, :))
df4 = DataFrame(hcat(vcat(breaks, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out5, priors5.diff, priors5.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap)
s1 = vcat(view(exp.(out5["Smp_trans"]), 1, :, :), view(exp.(extrap1), 1, :, :))
df5 = DataFrame(hcat(vcat(breaks, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out6, priors6.diff, priors6.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap)
s1 = vcat(view(exp.(out6["Smp_trans"]), 1, :, :), view(exp.(extrap1), 1, :, :))
df6 = DataFrame(hcat(vcat(breaks, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

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
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,6,6,6)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,NA)
p5 <- dat5 %>%
    pivot_longer(Mean:UCI) %>%
    ggplot(aes(x = Time, y = value, col = name,linetype = name)) + geom_step() +
    theme_classic() +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,6,6,6)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,NA)
p6 <- dat6 %>%
    pivot_longer(Mean:UCI) %>%
    ggplot(aes(x = Time, y = value, col = name,linetype = name)) + geom_step() +
    theme_classic() +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,6,6,6)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,NA)
plot_grid(p1,p2,p3,p4,p5,p6, nrow = 2)
ggsave($plotsdir("Colon","Colon2.pdf"), width = 8, height = 6)
"""

s1 = view(exp.(out7["Smp_trans"]), 1, :, :)
df1 = DataFrame(hcat(breaks, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
s1 = view(exp.(out8["Smp_trans"]), 1, :, :)
df2 = DataFrame(hcat(breaks, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
s1 = view(exp.(out9["Smp_trans"]), 1, :, :)
df3 = DataFrame(hcat(breaks, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
Random.seed!(1237)
breaks_extrap = collect(3.2:0.1:20)
extrap1 = barker_extrapolation(out7, priors7.diff, priors7.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap)
s1 = vcat(view(exp.(out7["Smp_trans"]), 1, :, :), view(exp.(extrap1), 1, :, :))
df4 = DataFrame(hcat(vcat(breaks, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out8, priors8.diff, priors8.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap)
s1 = vcat(view(exp.(out8["Smp_trans"]), 1, :, :), view(exp.(extrap1), 1, :, :))
df5 = DataFrame(hcat(vcat(breaks, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out9, priors9.diff, priors9.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap)
s1 = vcat(view(exp.(out9["Smp_trans"]), 1, :, :), view(exp.(extrap1), 1, :, :))
df6 = DataFrame(hcat(vcat(breaks, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

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
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,6,6,6)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,NA)
p5 <- dat5 %>%
    pivot_longer(Mean:UCI) %>%
    ggplot(aes(x = Time, y = value, col = name,linetype = name)) + geom_step() +
    theme_classic() +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,6,6,6)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,NA)
p6 <- dat6 %>%
    pivot_longer(Mean:UCI) %>%
    ggplot(aes(x = Time, y = value, col = name,linetype = name)) + geom_step() +
    theme_classic() +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,6,6,6)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,NA)
plot_grid(p1,p2,p3,p4,p5,p6, nrow = 2)
ggsave($plotsdir("Colon","Colon3.pdf"), width = 8, height = 6)
"""

grid = collect(0.01:0.01:3.2)
test_smp = cts_transform(cumsum(out10["Smp_x"], dims = 2), out10["Smp_s_loc"], grid)
s1 = view(exp.(test_smp), 1, :, :)
df1 = DataFrame(hcat(grid, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
test_smp = cts_transform(cumsum(out11["Smp_x"], dims = 2), out11["Smp_s_loc"], grid)
s1 = view(exp.(test_smp), 1, :, :)
df2 = DataFrame(hcat(grid, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
test_smp = cts_transform(cumsum(out12["Smp_x"], dims = 2), out12["Smp_s_loc"], grid)
s1 = view(exp.(test_smp), 1, :, :)
df3 = DataFrame(hcat(grid, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

Random.seed!(1237)
breaks_extrap = collect(3.2:0.1:15)
extrap1 = barker_extrapolation(out10, priors10.diff, priors10.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap)
test_smp = cts_transform(cumsum(out10["Smp_x"], dims = 2), out10["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), 1, :, :))
df4 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out11, priors11.diff, priors11.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap)
test_smp = cts_transform(cumsum(out11["Smp_x"], dims = 2), out11["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), 1, :, :))
df5 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out12, priors12.diff, priors12.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap)
test_smp = cts_transform(cumsum(out12["Smp_x"], dims = 2), out12["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), 1, :, :))
df6 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)


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
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,6,6,6)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,NA)
p5 <- dat5 %>%
    pivot_longer(Mean:UCI) %>%
    ggplot(aes(x = Time, y = value, col = name,linetype = name)) + geom_step() +
    theme_classic() +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,6,6,6)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,NA)
p6 <- dat6 %>%
    pivot_longer(Mean:UCI) %>%
    ggplot(aes(x = Time, y = value, col = name,linetype = name)) + geom_step() +
    theme_classic() +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,6,6,6)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,NA)
plot_grid(p1,p2,p3,p4,p5,p6, nrow = 2)
ggsave($plotsdir("Colon","Colon4.pdf"), width = 8, height = 6)
"""




grid = collect(0.01:0.01:3.2)
test_smp = cts_transform(cumsum(out13["Smp_x"], dims = 2), out13["Smp_s_loc"], grid)
s1 = view(exp.(test_smp), 1, :, :)
df1 = DataFrame(hcat(grid, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
test_smp = cts_transform(cumsum(out14["Smp_x"], dims = 2), out14["Smp_s_loc"], grid)
s1 = view(exp.(test_smp), 1, :, :)
df2 = DataFrame(hcat(grid, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
test_smp = cts_transform(cumsum(out15["Smp_x"], dims = 2), out15["Smp_s_loc"], grid)
s1 = view(exp.(test_smp), 1, :, :)
df3 = DataFrame(hcat(grid, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

Random.seed!(1237)
breaks_extrap = collect(3.3:0.1:15)
extrap1 = barker_extrapolation(out13, priors13.diff[1], priors13.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap)
test_smp = cts_transform(cumsum(out13["Smp_x"], dims = 2), out13["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), 1, :, :))
df4 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out14, priors14.diff[1], priors14.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap)
test_smp = cts_transform(cumsum(out14["Smp_x"], dims = 2), out14["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), 1, :, :))
df5 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out15, priors15.diff[1], priors15.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap)
test_smp = cts_transform(cumsum(out15["Smp_x"], dims = 2), out15["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), 1, :, :))
df6 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
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


grid = collect(0.01:0.01:3.2)
test_smp = cts_transform(cumsum(out16["Smp_x"],dims = 2), out16["Smp_s_loc"], grid)
s1 = view(exp.(test_smp), 1, :, :)
s2 = exp.(test_smp[1,:,:] .+ test_smp[2,:,:])
df1 = DataFrame(hcat(grid, median(s1, dims = 2), quantile.(eachrow(s1), 0.025),  quantile.(eachrow(s1), 0.975), median(s2, dims = 2), quantile.(eachrow(s2), 0.025),  quantile.(eachrow(s2), 0.975)), :auto)

test_smp = cts_transform(cumsum(out17["Smp_x"],dims = 2), out17["Smp_s_loc"], grid)
s1 = view(exp.(test_smp), 1, :, :)
s2 = exp.(test_smp[1,:,:] .+ test_smp[2,:,:])
df2 = DataFrame(hcat(grid, median(s1, dims = 2), quantile.(eachrow(s1), 0.025),  quantile.(eachrow(s1), 0.975), median(s2, dims = 2), quantile.(eachrow(s2), 0.025),  quantile.(eachrow(s2), 0.975)), :auto)

test_smp = cts_transform(cumsum(out18["Smp_x"],dims = 2), out18["Smp_s_loc"], grid)
s1 = view(exp.(test_smp), 1, :, :)
s2 = exp.(test_smp[1,:,:] .+ test_smp[2,:,:])
df3 = DataFrame(hcat(grid, median(s1, dims = 2), quantile.(eachrow(s1), 0.025),  quantile.(eachrow(s1), 0.975), median(s2, dims = 2), quantile.(eachrow(s2), 0.025),  quantile.(eachrow(s2), 0.975)), :auto)

Random.seed!(1237)
breaks_extrap = collect(3.2:0.1:15)
extrap1 = barker_extrapolation(out16, priors16.diff[1], priors16.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
extrap2 = barker_extrapolation(out16, priors16.diff[2], priors16.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 2)
test_smp = cts_transform(cumsum(out16["Smp_x"], dims = 2), out16["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
s2 = vcat(exp.(test_smp[1,:,:] .+ test_smp[2,:,:]), exp.(extrap1 .+ extrap2))
df4 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025),  quantile.(eachrow(s1), 0.975), median(s2, dims = 2), quantile.(eachrow(s2), 0.025),  quantile.(eachrow(s2), 0.975)), :auto)

extrap1 = barker_extrapolation(out17, priors17.diff[1], priors17.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
extrap2 = barker_extrapolation(out17, priors17.diff[2], priors17.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 2)
test_smp = cts_transform(cumsum(out17["Smp_x"], dims = 2), out17["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
s2 = vcat(exp.(test_smp[1,:,:] .+ test_smp[2,:,:]), exp.(extrap1 .+ extrap2))
df5 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025),  quantile.(eachrow(s1), 0.975), median(s2, dims = 2), quantile.(eachrow(s2), 0.025),  quantile.(eachrow(s2), 0.975)), :auto)

extrap1 = barker_extrapolation(out18, priors18.diff[1], priors18.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
extrap2 = barker_extrapolation(out18, priors18.diff[2], priors18.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 2)
test_smp = cts_transform(cumsum(out18["Smp_x"], dims = 2), out18["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
s2 = vcat(exp.(test_smp[1,:,:] .+ test_smp[2,:,:]), exp.(extrap1 .+ extrap2))
df6 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025),  quantile.(eachrow(s1), 0.975), median(s2, dims = 2), quantile.(eachrow(s2), 0.025),  quantile.(eachrow(s2), 0.975)), :auto)

R"""
dat1 = data.frame($df1)
colnames(dat1) <- c("Time","Median1","LCI1","UCI1","Median2","LCI2","UCI2") 
dat2 = data.frame($df2)
colnames(dat2) <- c("Time","Median1","LCI1","UCI1","Median2","LCI2","UCI2") 
dat3 = data.frame($df3)
colnames(dat3) <- c("Time","Median1","LCI1","UCI1","Median2","LCI2","UCI2") 
dat4 = data.frame($df4)
colnames(dat4) <- c("Time","Median1","LCI1","UCI1","Median2","LCI2","UCI2") 
dat5 = data.frame($df5)
colnames(dat5) <- c("Time","Median1","LCI1","UCI1","Median2","LCI2","UCI2") 
dat6 = data.frame($df6)
colnames(dat6) <- c("Time","Median1","LCI1","UCI1","Median2","LCI2","UCI2")  
"""

R"""    
p1 <- dat1 %>%
    pivot_longer(Median1:UCI2) %>%
    ggplot(aes(x = Time, y = value, col = name,linetype = name)) + geom_step() +
    theme_classic() +
    geom_vline(xintercept = 3) +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,6,7,6,7)]) +
    scale_linetype_manual(values = c("dotdash","dotdash", "solid", "solid","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,NA)
p2 <- dat2 %>%
    pivot_longer(Median1:UCI2) %>%
    ggplot(aes(x = Time, y = value, col = name,linetype = name)) + geom_step() +
    theme_classic() +
    geom_vline(xintercept = 3) +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,6,7,6,7)]) +
    scale_linetype_manual(values = c("dotdash","dotdash", "solid", "solid","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,NA)
p3 <- dat3 %>%
    pivot_longer(Median1:UCI2) %>%
    ggplot(aes(x = Time, y = value, col = name,linetype = name)) + geom_step() +
    theme_classic() +
    geom_vline(xintercept = 3) +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,6,7,6,7)]) +
    scale_linetype_manual(values = c("dotdash","dotdash", "solid", "solid","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,NA)
p4 <- dat4 %>%
    pivot_longer(Median1:UCI2) %>%
    ggplot(aes(x = Time, y = value, col = name,linetype = name)) + geom_step() +
    theme_classic() +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,6,7,6,7)]) +
    geom_vline(xintercept = 3.2, linetype = "dotted") +
    scale_linetype_manual(values = c("dotdash","dotdash", "solid", "solid","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,NA)
p5 <- dat5 %>%
    pivot_longer(Median1:UCI2) %>%
    ggplot(aes(x = Time, y = value, col = name,linetype = name)) + geom_step() +
    theme_classic() +
    geom_vline(xintercept = 3.2, linetype = "dotted") +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,6,7,6,7)]) +
    scale_linetype_manual(values = c("dotdash","dotdash", "solid", "solid","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,NA)
p6 <- dat6 %>%
    pivot_longer(Median1:UCI2) %>%
    ggplot(aes(x = Time, y = value, col = name,linetype = name)) + geom_step() +
    theme_classic() +
    geom_vline(xintercept = 3.2, linetype = "dotted") +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,6,7,6,7)]) +
    scale_linetype_manual(values = c("dotdash","dotdash", "solid", "solid","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,NA)
#plot_grid(p1,p2,p3, nrow = 1)
#plot_grid(p4,p5,p6, nrow = 1)
plot_grid(p1,p2,p3,p4,p5,p6, nrow = 2)
#ggsave($plotsdir("Colon","Colon_note3.pdf"), width = 8, height = 4)
"""


#### Plots for note

s1 = view(exp.(out1["Smp_trans"]), 1, :, :)
df1 = DataFrame(hcat(breaks, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
s1 = view(exp.(out4["Smp_trans"]), 1, :, :)
df2 = DataFrame(hcat(breaks, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
grid = collect(0.01:0.01:3.2)
test_smp = cts_transform(cumsum(out13["Smp_x"], dims = 2), out13["Smp_s_loc"], grid)
s1 = view(exp.(test_smp), 1, :, :)
df3 = DataFrame(hcat(grid, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

R"""
dat1 = data.frame($df1)
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI") 
dat2 = data.frame($df2)
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI") 
dat3 = data.frame($df3)
colnames(dat3) <- c("Time","Mean","LCI","Q1","Q4","UCI") 
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

plot_grid(p1,p2,p3, nrow = 1)
ggsave($plotsdir("Colon","Colon_note1.pdf"), width = 8, height = 4)
"""


grid = collect(0.01:0.01:3.2)
Random.seed!(1237)
breaks_extrap = collect(3.2:0.1:15)
extrap1 = barker_extrapolation(out13, priors13.diff, priors13.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap)
test_smp = cts_transform(cumsum(out13["Smp_x"], dims = 2), out13["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), 1, :, :))
df4 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out14, priors14.diff, priors14.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap)
test_smp = cts_transform(cumsum(out14["Smp_x"], dims = 2), out14["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), 1, :, :))
df5 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

extrap1 = barker_extrapolation(out15, priors15.diff, priors15.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap)
test_smp = cts_transform(cumsum(out15["Smp_x"], dims = 2), out15["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), 1, :, :))
df6 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
R"""
dat4 = data.frame($df4)
colnames(dat4) <- c("Time","Mean","LCI","Q1","Q4","UCI") 
dat5 = data.frame($df5)
colnames(dat5) <- c("Time","Mean","LCI","Q1","Q4","UCI") 
dat6 = data.frame($df6)
colnames(dat6) <- c("Time","Mean","LCI","Q1","Q4","UCI") 
"""

R"""    
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
plot_grid(p4,p5,p6, nrow = 1)
ggsave($plotsdir("Colon","Colon_note2.pdf"), width = 8, height = 4)
"""

out18["Smp_h"]

out15["Smp_h"]

histogram(out15["Smp_h"][1,:])
histogram(out18["Smp_h"][1,:])

testΣ = cumsum(out18["Smp_x"],dims = 2)
initial = testΣ[1,end,:]
for i in 1:size(initial,1)
    initial[i] = testΣ[1, findlast(isinf.(testΣ[1,:,i]) .== false), i]
end
histogram(exp.(initial))
quantile(exp.(initial), 0.975)

histogram(exp.(extrap1[1,:]))