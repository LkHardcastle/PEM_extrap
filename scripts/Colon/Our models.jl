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


priors1 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [10.0], [10.0]), 1.0, Cts(10.0, 50.0, 3.2), [RandomWalk()])
priors2 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [10.0], [10.0]), 1.0, Cts(10.0, 50.0, 3.2), [GaussLangevin(-1.0,1.0)])
priors3 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [10.0], [10.0]), 1.0, Cts(10.0, 50.0, 3.2), [GammaLangevin(0.5,2)])
priors4 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [10.0], [10.0]), 1.0, Cts(10.0, 50.0, 3.2), [GompertzBaseline(0.5)])
Random.seed!(9102)
@time out1 = pem_sample(state0, dat, priors1, settings)
@time out2 = pem_sample(state0, dat, priors2, settings)
@time out3 = pem_sample(state0, dat, priors3, settings)
@time out4 = pem_sample(state0, dat, priors4, settings)

Random.seed!(1237)
grid = collect(0.02:0.02:3.198)
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

### Covariates

Random.seed!(9102)
df = CSV.read(datadir("colon.csv"), DataFrame)
y = df.years
maximum(y)
n = length(y)
breaks = collect(0.1:0.1:3.1)
p = 1
cens = df.status
covar = fill(1.0, 1, n)
Lev = (df.rx .== "Lev")
FU = (df.rx .== "Lev+5FU")
covar = [covar; transpose(Lev); transpose(FU)]
dat = init_data(y, cens, covar, breaks)
maximum(y[intersect(findall(Lev .== 1), findall(cens .== 1))])
maximum(y[intersect(findall(FU .== 1), findall(cens .== 1))])
maximum(y[intersect(findall(cens .== 1))])
x0, v0, s0 = init_params(p, dat)
x0[2,:] = vcat(x0[2,1], zeros(size(breaks) .-1))
v0[2,:] = vcat(v0[2,1], 1.0, zeros(size(breaks) .-2))
s0[2,:] = vcat(s0[2,1], true, zeros(Int,size(breaks) .-2))
x0[3,:] = vcat(x0[3,1], zeros(size(breaks) .-1))
v0[3,:] = vcat(v0[3,1], 1.0, zeros(size(breaks) .-2))
s0[3,:] = vcat(s0[3,1], true, zeros(Int,size(breaks) .-2))
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 300_000
nsmp = 20_000
settings = Settings(nits, nsmp, 1_000_000, 1.0,0.1, 0.5, false, true)


priors1 = BasicPrior(1.0, PC([0.2, 0.2, 0.2], [2, 2, 2], [0.5, 0.5, 0.5], Inf), Beta([0.4, 0.4, 0.4], [10.0, 5.0, 5.0], [10.0, 15.0, 15.0]), 1.0, Cts(10.0, 150.0, 3.2), [RandomWalk(), GaussLangevin(-0.5,0.5), GaussLangevin(-1,0.5)])
priors2 = BasicPrior(1.0, PC([0.2, 0.2, 0.2], [2, 2, 2], [0.5, 0.5, 0.5], Inf), Beta([0.4, 0.4, 0.4], [10.0, 5.0, 5.0], [10.0, 15.0, 15.0]), 1.0, Cts(10.0, 150.0, 3.2), [GaussLangevin(-1.0,1.0), GaussLangevin(-0.5,0.5), GaussLangevin(-1,0.5)])
priors3 = BasicPrior(1.0, PC([0.2, 0.2, 0.2], [2, 2, 2], [0.5, 0.5, 0.5], Inf), Beta([0.4, 0.4, 0.4], [10.0, 5.0, 5.0], [10.0, 15.0, 15.0]), 1.0, Cts(10.0, 150.0, 3.2), [GammaLangevin(0.5,2), GaussLangevin(-0.5,0.5), GaussLangevin(-1,0.5)])
priors4 = BasicPrior(1.0, PC([0.2, 0.2, 0.2], [2, 2, 2], [0.5, 0.5, 0.5], Inf), Beta([0.4, 0.4, 0.4], [10.0, 5.0, 5.0], [10.0, 15.0, 15.0]), 1.0, Cts(10.0, 150.0, 3.2), [GompertzBaseline(0.5), GaussLangevin(-0.5,0.5), GaussLangevin(-1,0.5)])
Random.seed!(9102)
@time out1 = pem_sample(state0, dat, priors1, settings)
@time out2 = pem_sample(state0, dat, priors2, settings)
@time out3 = pem_sample(state0, dat, priors3, settings)
@time out4 = pem_sample(state0, dat, priors4, settings)


Random.seed!(1237)
grid = collect(0.02:0.02:3.198)
breaks_extrap = collect(3.2:0.02:15)
extrap1 = barker_extrapolation(out1, priors1.diff[1], priors1.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
extrap2 = barker_extrapolation(out1, priors1.diff[2], priors1.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 2)
extrap3 = barker_extrapolation(out1, priors1.diff[3], priors1.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 3)
test_smp = cts_transform(cumsum(out1["Smp_x"], dims = 2), out1["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
s2 = vcat(exp.(test_smp[1,:,:] .+ test_smp[2,:,:]), exp.(extrap1 .+ extrap2))
s3 = vcat(exp.(test_smp[1,:,:] .+ test_smp[3,:,:]), exp.(extrap1 .+ extrap3))
df1 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025),  quantile.(eachrow(s1), 0.975), median(s2, dims = 2), quantile.(eachrow(s2), 0.025),  quantile.(eachrow(s2), 0.975), median(s3, dims = 2), quantile.(eachrow(s3), 0.025),  quantile.(eachrow(s3), 0.975)), :auto)

breaks_extrap = collect(3.2:0.02:15)
extrap1 = barker_extrapolation(out2, priors2.diff[1], priors2.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
extrap2 = barker_extrapolation(out2, priors2.diff[2], priors2.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 2)
extrap3 = barker_extrapolation(out2, priors2.diff[3], priors2.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 3)
test_smp = cts_transform(cumsum(out2["Smp_x"], dims = 2), out2["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
s2 = vcat(exp.(test_smp[1,:,:] .+ test_smp[2,:,:]), exp.(extrap1 .+ extrap2))
s3 = vcat(exp.(test_smp[1,:,:] .+ test_smp[3,:,:]), exp.(extrap1 .+ extrap3))
df2 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025),  quantile.(eachrow(s1), 0.975), median(s2, dims = 2), quantile.(eachrow(s2), 0.025),  quantile.(eachrow(s2), 0.975), median(s3, dims = 2), quantile.(eachrow(s3), 0.025),  quantile.(eachrow(s3), 0.975)), :auto)

breaks_extrap = collect(3.2:0.02:15)
extrap1 = barker_extrapolation(out3, priors3.diff[1], priors3.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
extrap2 = barker_extrapolation(out3, priors3.diff[2], priors3.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 2)
extrap3 = barker_extrapolation(out3, priors3.diff[3], priors3.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 3)
test_smp = cts_transform(cumsum(out3["Smp_x"], dims = 2), out3["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
s2 = vcat(exp.(test_smp[1,:,:] .+ test_smp[2,:,:]), exp.(extrap1 .+ extrap2))
s3 = vcat(exp.(test_smp[1,:,:] .+ test_smp[3,:,:]), exp.(extrap1 .+ extrap3))
df3 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025),  quantile.(eachrow(s1), 0.975), median(s2, dims = 2), quantile.(eachrow(s2), 0.025),  quantile.(eachrow(s2), 0.975), median(s3, dims = 2), quantile.(eachrow(s3), 0.025),  quantile.(eachrow(s3), 0.975)), :auto)

breaks_extrap = collect(3.2:0.02:15)
extrap1 = barker_extrapolation(out4, priors4.diff[1], priors4.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
extrap2 = barker_extrapolation(out4, priors4.diff[2], priors4.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 2)
extrap3 = barker_extrapolation(out4, priors4.diff[3], priors4.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 3)
test_smp = cts_transform(cumsum(out4["Smp_x"], dims = 2), out4["Smp_s_loc"], grid)
s1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
s2 = vcat(exp.(test_smp[1,:,:] .+ test_smp[2,:,:]), exp.(extrap1 .+ extrap2))
s3 = vcat(exp.(test_smp[1,:,:] .+ test_smp[3,:,:]), exp.(extrap1 .+ extrap3))
df4 = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1, dims = 2), quantile.(eachrow(s1), 0.025),  quantile.(eachrow(s1), 0.975), median(s2, dims = 2), quantile.(eachrow(s2), 0.025),  quantile.(eachrow(s2), 0.975), median(s3, dims = 2), quantile.(eachrow(s3), 0.025),  quantile.(eachrow(s3), 0.975)), :auto)



R"""
dat1 = data.frame($df1)
colnames(dat1) <- c("Time","Median1","LCI1","UCI1","Median2","LCI2","UCI2","Median3","LCI3","UCI3") 
dat2 = data.frame($df2)
colnames(dat2) <- c("Time","Median1","LCI1","UCI1","Median2","LCI2","UCI2","Median3","LCI3","UCI3") 
dat3 = data.frame($df3)
colnames(dat3) <- c("Time","Median1","LCI1","UCI1","Median2","LCI2","UCI2","Median3","LCI3","UCI3") 
dat4 = data.frame($df4)
colnames(dat4) <- c("Time","Median1","LCI1","UCI1","Median2","LCI2","UCI2","Median3","LCI3","UCI3") 
"""

R"""    
p1 <- dat1 %>%
    pivot_longer(Median1:UCI3) %>%
    ggplot(aes(x = Time, y = value, col = name,linetype = name)) + geom_step() +
    theme_classic() +
    geom_vline(xintercept = 3) +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,4,6,7,4,6,7,4)]) +
    scale_linetype_manual(values = c("dotdash","dotdash","dotdash","solid", "solid", "solid","dotdash","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,1) + xlim(0,15)
p2 <- dat2 %>%
    pivot_longer(Median1:UCI3) %>%
    ggplot(aes(x = Time, y = value, col = name,linetype = name)) + geom_step() +
    theme_classic() +
    geom_vline(xintercept = 3) +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,4,6,7,4,6,7,4)]) +
    scale_linetype_manual(values = c("dotdash","dotdash","dotdash","solid", "solid", "solid","dotdash","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,1) + xlim(0,15)
p3 <- dat3 %>%
    pivot_longer(Median1:UCI3) %>%
    ggplot(aes(x = Time, y = value, col = name,linetype = name)) + geom_step() +
    theme_classic() +
    geom_vline(xintercept = 3) +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,4,6,7,4,6,7,4)]) +
    scale_linetype_manual(values = c("dotdash","dotdash","dotdash","solid", "solid", "solid","dotdash","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,1) + xlim(0,15)
p4 <- dat4 %>%
    pivot_longer(Median1:UCI3) %>%
    ggplot(aes(x = Time, y = value, col = name,linetype = name)) + geom_step() +
    theme_classic() +
    geom_vline(xintercept = 3) +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,4,6,7,4,6,7,4)]) +
    scale_linetype_manual(values = c("dotdash","dotdash","dotdash","solid", "solid", "solid","dotdash","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,1) + xlim(0,15)
plot_grid(p1,p2,p3,p4, nrow = 2)
ggsave($plotsdir("CovariateColon.pdf"), width = 8, height = 6)
"""
