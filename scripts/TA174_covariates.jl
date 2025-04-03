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
breaks = vcat(0.01,collect(0.25:0.1:4.1))
p = 1
cens = df.death
covar = fill(1.0, 1, n)
trt = (df.treat .== 1)
covar = [covar; transpose(trt)]
dat = init_data(y, cens, covar, breaks)
t0 = 0.0
x0, v0, s0 = init_params(p, dat)
x0[2,:] = vcat(x0[2,1], zeros(size(breaks) .-1))
v0[2,:] = vcat(v0[2,1], 1.0, zeros(size(breaks) .-2))
s0[2,:] = vcat(s0[2,1], true, zeros(Int,size(breaks) .-2))
v0 = v0./norm(v0)
nits = 10_000
burn_in = 5_000
nsmp = 10
test_times = [0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0]
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
priors = BasicPrior(1.0, PC(1.0, 2, 0.5, Inf), FixedW([0.5,0.5]), 1.0, CtsPois(15.0, 1.0, 100.0, 4.1), [GammaLangevin(5,15,1), GaussLangevin(t -> 0.0, t -> 1.0)], [0.01], 2)
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 50.0)
@time out1 = pem_fit(state0, dat, priors, settings, test_times, burn_in)
println(out1[3]);println(out1[4])

grid = sort(unique(out1[1]["Sk_s_loc"][cumsum(out1[1]["Sk_s"],dims = 1)[1,:,:] .> 0.0]))
grid = grid[1:1:length(grid)]
test_smp = cts_transform(cumsum(out1[1]["Sk_θ"], dims = 2), out1[1]["Sk_s_loc"], grid)[:,:,burn_in:nits]

s1 = view(exp.(test_smp), 1, :, :)
df1 = DataFrame(hcat(grid, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
s2 = exp.(view(test_smp, 1, :, :) .+ view(test_smp, 2, :, :))
df2 = DataFrame(hcat(grid, median(s2, dims = 2), quantile.(eachrow(s2), 0.025), quantile.(eachrow(s2), 0.25), quantile.(eachrow(s2), 0.75), quantile.(eachrow(s2), 0.975)), :auto)

Random.seed!(2352)
breaks_extrap = collect(4.12:0.02:15)
priors = BasicPrior(1.0, PC(1.0, 2, 0.5, Inf), FixedW([0.5,0.5]), 1.0, CtsPois(15.0, 1.0, 100.0, 4.1), [GammaLangevin(5,15,1), GaussLangevin(t -> 0.0, t -> 1.0)], [0.01], 2)
extrap1 = barker_extrapolation(out1[1], priors.diff[1], priors.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1, 0.1)
extrap2 = barker_extrapolation(out1[1], priors.diff[2], priors.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 2, 0.1)
priors = BasicPrior(1.0, PC(1.0, 2, 0.5, Inf), FixedW([0.5,0.5]), 1.0, CtsPois(15.0, 1.0, 100.0, 4.1), [GammaLangevin(5,15,1), GaussLangevin(t -> 0.0, t -> (max(1,t^2/16)^-1))], [0.01], 2)
extrap3 = barker_extrapolation(out1[1], priors.diff[2], priors.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 2, 0.1)
s1_ = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, burn_in:nits))
s2_ = vcat(exp.(view(test_smp, 1, :, :) .+ view(test_smp, 2, :, :)), exp.(view(extrap1, :, burn_in:nits) .+ view(extrap2, :, burn_in:nits)))
s3_ = vcat(exp.(view(test_smp, 1, :, :) .+ view(test_smp, 2, :, :)), exp.(view(extrap1, :, burn_in:nits) .+ view(extrap3, :, burn_in:nits)))

m1 = get_meansurv(log.(s1), grid, [1])
m2 = get_meansurv(log.(s2), grid, [1])
mean(m1)
mean(m2)
quantile(m1, [0.025,0.975])
quantile(m2, [0.025,0.975])
mean(m2 .- m1)
quantile(m2 .- m1, [0.025,0.975])

m1_ = get_meansurv(log.(s1_), vcat(grid, breaks_extrap), [1])
m2_ = get_meansurv(log.(s2_), vcat(grid, breaks_extrap), [1])
m3_ = get_meansurv(log.(s3_), vcat(grid, breaks_extrap), [1])
mean(m1_)
mean(m2_)
mean(m3_)
quantile(m1_, [0.025,0.975])
quantile(m2_, [0.025,0.975])
quantile(m3_, [0.025,0.975])
mean(m2_ .- m1_)
quantile(m2_ .- m1_, [0.025,0.975])
mean(m3_ .- m1_)
quantile(m3_ .- m1_, [0.025,0.975])

df1_ = DataFrame(hcat(vcat(grid, breaks_extrap), median(s1_, dims = 2), quantile.(eachrow(s1_), 0.025), quantile.(eachrow(s1_), 0.25), quantile.(eachrow(s1_), 0.75), quantile.(eachrow(s1_), 0.975)), :auto)
df2_ = DataFrame(hcat(vcat(grid, breaks_extrap), median(s2_, dims = 2), quantile.(eachrow(s2_), 0.025), quantile.(eachrow(s2_), 0.25), quantile.(eachrow(s2_), 0.75), quantile.(eachrow(s2_), 0.975)), :auto)
df3_ = DataFrame(hcat(vcat(grid, breaks_extrap), median(s3_, dims = 2), quantile.(eachrow(s3_), 0.025), quantile.(eachrow(s3_), 0.25), quantile.(eachrow(s3_), 0.75), quantile.(eachrow(s3_), 0.975)), :auto)

CSV.write(datadir("TA174Models","CovCtrl.csv"),df1_)
CSV.write(datadir("TA174Models","CovTrt.csv"),df2_)
CSV.write(datadir("TA174Models","CovWane.csv"),df3_)

R"""
dat1 = data.frame($df1_)
dat1$Arm = "Placebo"
dat2 = data.frame($df2_)
dat2$Arm = "Treatment"
dat1 = rbind(dat1, dat2)
dat3 = data.frame($df3_)
dat3$Arm = "Treatment (Waning)"
dat3 = rbind(dat1,dat3)
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Arm") 
colnames(dat3) <- c("Time","Mean","LCI","Q1","Q4","UCI","Arm") 
p1 <- dat1 %>%
    subset(Time < 4.1) %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Arm, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.3) + xlim(0.01,4) 
p2 <- dat3 %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    subset(Time > 4.0) %>%
    ggplot(aes(x = Time, y = log(value), col = Arm, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,4)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + xlim(4.01,NA) #+ ylim(0,2)
    plot_grid(p1,p2)
"""