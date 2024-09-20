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
library(ggsurvfit)
library(survival)
cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
"""



####### Treatment trial -

Random.seed!(9102)
df_all = CSV.read(datadir("SOLVD","SOLVD.csv"), DataFrame)
df = filter(idx -> idx.DRUG == "P" && idx.TRIAL == "T" && idx.EPYTIME > 0.0, df_all)
sum(df.EPX)
y = df.EPYTIME/365
maximum(y)
minimum(y)
n = length(y)
breaks = collect(0.02:0.1:4.62)
p = 1
cens = df.EPX
sum(cens)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 100_000
nsmp = 20000
settings = Settings(nits, nsmp, 1_000_000, 1.0,0.5, 0.5, false, true)

priors1 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [10.0], [10.0]), 1.0, Cts(5.0, 150.0, 4.7), [RandomWalk()])
priors2 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [10.0], [10.0]), 1.0, Cts(5.0, 150.0, 4.7), [GaussLangevin(-1.0,1.0)])
priors3 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [10.0], [10.0]), 1.0, Cts(5.0, 150.0, 4.7), [GammaLangevin(0.5,2)])
priors4 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [10.0], [10.0]), 1.0, Cts(5.0, 150.0, 4.7), [GompertzBaseline(0.5)])
Random.seed!(9102)
@time out1 = pem_sample(state0, dat, priors1, settings)
@time out2 = pem_sample(state0, dat, priors2, settings)
@time out3 = pem_sample(state0, dat, priors3, settings)
@time out4 = pem_sample(state0, dat, priors4, settings)

Random.seed!(1237)
grid = collect(0.02:0.02:4.698)
breaks_extrap = collect(4.7:0.02:15)
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

histogram(filter(idx -> !isinf(idx), vec(out1["Smp_s_loc"])))

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
dat_diffusion1 <- rbind(dat1, dat2, dat3, dat4)
"""

R"""
p1 <- dat_diffusion1 %>%
    subset(Time < 5.2) %>%
    #subset(Model == "Random Walk")  %>%
    pivot_longer(Mean:UCI,) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,1) + xlim(0,5.2)
p2 <- dat_diffusion1 %>%
    #subset(Time < 5.2) %>%
    #subset(Model == "Gamma Langevin")  %>%
    pivot_longer(Mean:UCI,) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,5) + xlim(0,15) +
    geom_vline(aes(xintercept = 5.2), linetype = "dotted")
plot_grid(p1,p2)
"""


Random.seed!(9102)
df_all = CSV.read(datadir("SOLVD","SOLVD.csv"), DataFrame)
df = filter(idx -> idx.DRUG == "D" && idx.TRIAL == "T" && idx.EPYTIME > 0.0, df_all)
sum(df.EPX)
y = df.EPYTIME/365
maximum(y)
minimum(y)
n = length(y)
breaks = collect(0.02:0.1:4.62)
p = 1
cens = df.EPX
sum(cens)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 100_000
nsmp = 20000
settings = Settings(nits, nsmp, 1_000_000, 1.0,0.5, 0.5, false, true)

priors1 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [10.0], [10.0]), 1.0, Cts(5.0, 150.0, 4.7), [RandomWalk()])
priors2 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [10.0], [10.0]), 1.0, Cts(5.0, 150.0, 4.7), [GaussLangevin(-1.0,1.0)])
priors3 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [10.0], [10.0]), 1.0, Cts(5.0, 150.0, 4.7), [GammaLangevin(0.5,2)])
priors4 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [10.0], [10.0]), 1.0, Cts(5.0, 150.0, 4.7), [GompertzBaseline(0.5)])
Random.seed!(9102)
@time out1 = pem_sample(state0, dat, priors1, settings)
@time out2 = pem_sample(state0, dat, priors2, settings)
@time out3 = pem_sample(state0, dat, priors3, settings)
@time out4 = pem_sample(state0, dat, priors4, settings)

Random.seed!(1237)
grid = collect(0.02:0.02:4.698)
breaks_extrap = collect(4.7:0.02:15)
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

histogram(filter(idx -> !isinf(idx), vec(out1["Smp_s_loc"])))

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
dat_diffusion2 <- rbind(dat1, dat2, dat3, dat4)
"""

R"""
p1 <- dat_diffusion2 %>%
    subset(Time < 5.2) %>%
    #subset(Model == "Random Walk")  %>%
    pivot_longer(Mean:UCI,) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,1) + xlim(0,5.2)
p2 <- dat_diffusion2 %>%
    #subset(Time < 5.2) %>%
    #subset(Model == "Gamma Langevin")  %>%
    pivot_longer(Mean:UCI,) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,5) + xlim(0,15) +
    geom_vline(aes(xintercept = 5.2), linetype = "dotted")
plot_grid(p1,p2)
"""

R"""
dat_diffusion1$Trt <- "P"
dat_diffusion2$Trt <- "D"
dat_diffusion <- rbind(dat_diffusion1, dat_diffusion2)
"""

R"""
p1 <- dat_diffusion %>%
    subset(Time < 5.2) %>%
    subset(Model == "Gamma Langevin")  %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Trt, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,1) + xlim(0,5.2)
p2 <- dat_diffusion %>%
    #subset(Time < 5.2) %>%
    subset(Model == "Gamma Langevin")  %>%
    pivot_longer(c(Mean, LCI, UCI),) %>%
    ggplot(aes(x = Time, y = value, col = Trt, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,1.5) + xlim(0,15) +
    geom_vline(aes(xintercept = 5.2), linetype = "dotted")
plot_grid(p1,p2)
"""

Random.seed!(9102)
df_all = CSV.read(datadir("SOLVD","SOLVD.csv"), DataFrame)
df = filter(idx -> idx.TRIAL == "T" && idx.EPYTIME > 0.0, df_all)
sum(df.EPX)
y = df.EPYTIME/365
maximum(y)
minimum(y)
n = length(y)
breaks = collect(0.02:0.1:4.62)
p = 1
cens = df.EPX
covar = fill(1.0, 1, n)
trt = (df.DRUG .== "P").*1.0
covar = [covar; transpose(trt)]
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
x0[2,:] = vcat(x0[2,1], zeros(size(breaks) .-1))
v0[2,:] = vcat(v0[2,1], 1.0, zeros(size(breaks) .-2))
s0[2,:] = vcat(s0[2,1], true, zeros(Int,size(breaks) .-2))

v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 500_000
nsmp = 20_000
settings = Settings(nits, nsmp, 1_000_000, 1.0,0.1, 0.5, false, true)

-log(0.35)

a = 5.5
b = 3.6
a/b
-log(0.35)
quantile(Gamma(a, 1/b),0.25)
quantile(Gamma(a, 1/b),0.05)
quantile(Gamma(a, 1/b),0.95)
exp(-2.7)
exp(1)
exp(-1)
priors1 = BasicPrior(1.0, PC([0.2, 0.2], [2, 2], [0.5, 0.5], Inf), Beta([0.4, 0.4], [10.0, 10.0], [10.0, 10.0]), 1.0, Cts(15.0, 200.0, 4.7), [GammaLangevin(a,b), GaussLangevin(0.0,0.5)])
Random.seed!(9102)
@time out1 = pem_sample(state0, dat, priors1, settings)

histogram(out1["Smp_J"])
histogram(vec(sum(out1["Smp_s"][1,:,:], dims = 1)))
histogram!(vec(sum(out1["Smp_s"][2,:,:] , dims = 1).+ 0.5))
maximum(y)

Random.seed!(1237)
grid = sort(unique(out1["Smp_s_loc"][cumsum(out1["Smp_s"],dims = 1)[2,:,:] .> 0.0]))
grid = grid[1:10:length(grid)]

breaks_extrap = collect(4.7:0.02:15)
extrap1 = barker_extrapolation(out1, priors1.diff[1], priors1.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
extrap2 = barker_extrapolation(out1, priors1.diff[2], priors1.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 2)
test_smp = cts_transform(cumsum(out1["Smp_x"], dims = 2), out1["Smp_s_loc"], grid)
h1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
h2 = vcat(exp.(test_smp[1,:,:] .+ test_smp[2,:,:]), exp.(extrap1 .+ extrap2))
df1 = DataFrame(hcat(vcat(grid, breaks_extrap), median(h1, dims = 2), quantile.(eachrow(h1), 0.025),  quantile.(eachrow(h1), 0.975), median(h2, dims = 2), quantile.(eachrow(h2), 0.025),  quantile.(eachrow(h2), 0.975)), :auto)

s1 = pem_survival(h1, vcat(0.0, grid, breaks_extrap))
s2 = pem_survival(h2, vcat(0.0, grid, breaks_extrap))
df2 = DataFrame(hcat(vcat(0.0, grid, breaks_extrap)[1:(end-1)], median(s1, dims = 2), quantile.(eachrow(s1), 0.025),  quantile.(eachrow(s1), 0.975), median(s2, dims = 2), quantile.(eachrow(s2), 0.025),  quantile.(eachrow(s2), 0.975)), :auto)

hr = h2./h1

df3 = DataFrame(hcat(vcat(grid, breaks_extrap), median(hr, dims = 2), quantile.(eachrow(hr), 0.025),  quantile.(eachrow(hr), 0.975)), :auto)

R"""
dat1 = data.frame($df1)
colnames(dat1) <- c("Time","Median1","LCI1","UCI1","Median2","LCI2","UCI2")
dat2 = data.frame($df2)
colnames(dat2) <- c("Time","Median1","LCI1","UCI1","Median2","LCI2","UCI2") 
dat3 = data.frame($df3)
colnames(dat3) <- c("Time","Median1","LCI1","UCI1") 
"""

R"""    
p1 <- dat1 %>%
    pivot_longer(Median1:UCI2) %>%
    mutate(Arm = case_when(
        grepl("1", name, fixed = TRUE) ~ "Placebo",
        grepl("2", name, fixed = TRUE) ~ "Enalapril",
            ),
        Stat = case_when(
        grepl("Median", name, fixed = TRUE) ~ "Median",
        grepl("UCI", name, fixed = TRUE) ~ "UCI",
        grepl("LCI", name, fixed = TRUE) ~ "LCI",
            )
    ) %>%
    ggplot(aes(x = Time, y = value, col = Arm, linetype = Stat)) + geom_step() +
    theme_classic() +
    geom_vline(xintercept = 4.7) +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid",  "dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,1) + xlim(0,4.6)
#ggsave($plotsdir("CovariateColon.pdf"), width = 8, height = 6)
p2 <- dat1 %>%
    pivot_longer(Median1:UCI2) %>%
    mutate(Arm = case_when(
        grepl("1", name, fixed = TRUE) ~ "Placebo",
        grepl("2", name, fixed = TRUE) ~ "Enalapril",
            ),
        Stat = case_when(
        grepl("Median", name, fixed = TRUE) ~ "Median",
        grepl("UCI", name, fixed = TRUE) ~ "UCI",
        grepl("LCI", name, fixed = TRUE) ~ "LCI",
            )
    ) %>%
    ggplot(aes(x = Time, y = value, col = Arm, linetype = Stat)) + geom_step() +
    theme_classic() +
    geom_vline(xintercept = 4.7) +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid", "dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,1) + xlim(0,15)

km = survfit(Surv($y,$cens) ~ $trt)
time1 = km$time[1:km$strata[1]]
time2 = km$time[(km$strata[1] + 1):length(km$time)]
surv1 = km$surv[1:km$strata[1]]
surv2 = km$surv[(km$strata[1] + 1):length(km$surv)]
km_dat1 = data.frame(time = time1, surv = surv1)
km_dat2 = data.frame(time = time2, surv = surv2)

p3 <- dat2 %>%
    pivot_longer(Median1:UCI2) %>%
    mutate(Arm = case_when(
        grepl("1", name, fixed = TRUE) ~ "Placebo",
        grepl("2", name, fixed = TRUE) ~ "Enalapril",
            ),
        Stat = case_when(
        grepl("Median", name, fixed = TRUE) ~ "Median",
        grepl("UCI", name, fixed = TRUE) ~ "UCI",
        grepl("LCI", name, fixed = TRUE) ~ "LCI",
            )
    ) %>%
    ggplot(aes(x = Time, y = value)) + geom_step(aes(col = Arm, linetype = Stat)) + 
    theme_classic() + 
    geom_vline(xintercept = 4.7) +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid", "dotdash"))+ ylab("S(t)") + xlab("Time (years)") + ylim(0,1) + xlim(0,4.6)
p3 <- p3 + geom_step(data = km_dat1, aes(x = time, y = surv), col = cbPalette[7]) + geom_step(data = km_dat2, aes(x = time, y = surv), col = cbPalette[6])
p4 <- dat2 %>%
    pivot_longer(Median1:UCI2) %>%
    mutate(Arm = case_when(
        grepl("1", name, fixed = TRUE) ~ "Placebo",
        grepl("2", name, fixed = TRUE) ~ "Enalapril",
            ),
        Stat = case_when(
        grepl("Median", name, fixed = TRUE) ~ "Median",
        grepl("UCI", name, fixed = TRUE) ~ "UCI",
        grepl("LCI", name, fixed = TRUE) ~ "LCI",
            )
    ) %>%
    ggplot(aes(x = Time, y = value)) + geom_step(aes(col = Arm, linetype = Stat)) + 
    theme_classic() +
    geom_vline(xintercept = 4.7, linetype = "dotted") +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid", "dotdash")) + ylab("S(t)") + xlab("Time (years)") + ylim(0,1) + xlim(0,15)
p4 <- p4 + geom_step(data = km_dat1, aes(x = time, y = surv), col = cbPalette[7]) + geom_step(data = km_dat2, aes(x = time, y = surv), col = cbPalette[6])

p5 <- dat3 %>%
    pivot_longer(Median1:UCI1) %>%
    ggplot(aes(x = Time, y = value, linetype = name)) + geom_step(col = cbPalette[7]) + 
    theme_classic() +
    geom_vline(xintercept = 4.7, linetype = "dotted") +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid", "dotdash")) + ylab("Hazard ratio") + xlab("Time (years)") + ylim(0,2.5) + xlim(0,15)
p5
plot_grid(p1,p2,p4, p5)
ggsave($plotsdir("SOLVD_Trt.pdf"), width = 8, height = 4)
"""
