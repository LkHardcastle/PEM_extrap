using DrWatson
@quickactivate "PEM_extrap"
# For src
using DataStructures, LinearAlgebra, Distributions, Random, Optim, Roots, SpecialFunctions
using Plots, CSV, DataFrames, RCall, Interpolations, JLD2

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

#### Look at development of CHF

Random.seed!(9102)
df_all = CSV.read(datadir("SOLVD","SOLVD.csv"), DataFrame)
df = filter(idx -> idx.TRIAL == "P" && idx.EPATIME > 0.0, df_all)
sum(df.EPA)
y = df.EPATIME/365
maximum(y)
minimum(y)
n = length(y)
breaks = vcat(0.001, collect(0.02:0.1:5.22))
p = 1
cens = df.EPA
covar = fill(1.0, 1, n)
trt = (df.DRUG .== "P").*1.0
covar = [covar; transpose(trt)]
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
x0[2,:] = vcat(x0[2,1], zeros(size(breaks) .-1))
v0[2,:] = vcat(v0[2,1], 1.0, zeros(size(breaks) .-2))
s0[2,:] = vcat(s0[2,1], true, zeros(Int,size(breaks) .-2))

#### Interval censoring - don't investigate further

R"""
km = survfit(Surv($y,$cens) ~ $trt)
time1 = km$time[1:km$strata[1]]
time2 = km$time[(km$strata[1] + 1):length(km$time)]
surv1 = km$surv[1:km$strata[1]]
surv2 = km$surv[(km$strata[1] + 1):length(km$surv)]
km_dat1 = data.frame(time = time1, surv = surv1)
km_dat2 = data.frame(time = time2, surv = surv2)
p <- ggplot() + 
    theme_classic() +
    geom_step(data = km_dat1, aes(x = time, y = surv), col = cbPalette[7]) + 
    geom_step(data = km_dat2, aes(x = time, y = surv), col = cbPalette[6]) +
    geom_vline(xintercept = seq(0.3333, 6, 0.3333), linetype = "dashed")
"""


Random.seed!(9102)
df_all = CSV.read(datadir("SOLVD","SOLVD.csv"), DataFrame)
df = filter(idx -> idx.TRIAL == "P" && idx.EPYTIME > 0.0, df_all)
sum(df.EPY)
y = df.EPYTIME/365
maximum(y)
minimum(y)
n = length(y)
breaks = vcat(0.001, collect(0.02:0.1:5.22))
p = 1
cens = df.EPY
covar = fill(1.0, 1, n)
trt = (df.DRUG .== "P").*1.0
covar = [covar; transpose(trt)]
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
x0[2,:] = vcat(x0[2,1], zeros(size(breaks) .-1))
v0[2,:] = vcat(v0[2,1], 1.0, zeros(size(breaks) .-2))
s0[2,:] = vcat(s0[2,1], true, zeros(Int,size(breaks) .-2))

R"""
km = survfit(Surv($y,$cens) ~ $trt)
time1 = km$time[1:km$strata[1]]
time2 = km$time[(km$strata[1] + 1):length(km$time)]
surv1 = km$surv[1:km$strata[1]]
surv2 = km$surv[(km$strata[1] + 1):length(km$surv)]
km_dat1 = data.frame(time = time1, surv = surv1)
km_dat2 = data.frame(time = time2, surv = surv2)
p <- ggplot() + 
    theme_classic() +
    geom_step(data = km_dat1, aes(x = time, y = surv), col = cbPalette[7]) + 
    geom_step(data = km_dat2, aes(x = time, y = surv), col = cbPalette[6]) + ylim(0,NA)
"""


# Based on figures from the paper
-log(0.85)
-log(0.5)
μ = -1.35
σ = 0.6
quantile(LogNormal(μ, σ),0.25)
quantile(LogNormal(μ, σ),0.5)
quantile(LogNormal(μ, σ),0.95)


v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 1_000_000
nsmp = 20_000
settings = Settings(nits, nsmp, 1_000_000, 1.0,0.1, 0.5, false, true)

priors1 = BasicPrior(1.0, PC([0.2, 0.2], [2, 2], [0.5, 0.5], Inf), Beta([0.4, 0.4], [10.0, 10.0], [10.0, 10.0]), 1.0, Cts(15.0, 250.0, 5.3), [GaussLangevin(μ,σ), GaussLangevin(0.0,0.5)])
Random.seed!(9102)
@time out1 = pem_sample(state0, dat, priors1, settings)

Random.seed!(1237)
grid = sort(unique(out1["Smp_s_loc"][cumsum(out1["Smp_s"],dims = 1)[2,:,:] .> 0.0]))
grid = grid[1:10:length(grid)]

breaks_extrap = collect(5.3:0.02:15)
extrap1 = barker_extrapolation(out1, priors1.diff[1], priors1.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
extrap2 = barker_extrapolation(out1, priors1.diff[2], priors1.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 2)

extrap2[1,:]
test_smp = cts_transform(cumsum(out1["Smp_x"], dims = 2), out1["Smp_s_loc"], grid)
h1 = vcat(view(exp.(test_smp), 1, :, :), view(exp.(extrap1), :, :))
h2 = vcat(exp.(test_smp[1,:,:] .+ test_smp[2,:,:]), exp.(extrap1 .+ extrap2))
df1 = DataFrame(hcat(vcat(grid, breaks_extrap), median(h1, dims = 2), quantile.(eachrow(h1), 0.025),  quantile.(eachrow(h1), 0.975), median(h2, dims = 2), quantile.(eachrow(h2), 0.025),  quantile.(eachrow(h2), 0.975)), :auto)

s1 = pem_survival(h1, vcat(0.0, grid, breaks_extrap))
s2 = pem_survival(h2, vcat(0.0, grid, breaks_extrap))
df2 = DataFrame(hcat(vcat(0.0, grid, breaks_extrap)[1:(end-1)], median(s1, dims = 2), quantile.(eachrow(s1), 0.025),  quantile.(eachrow(s1), 0.975), median(s2, dims = 2), quantile.(eachrow(s2), 0.025),  quantile.(eachrow(s2), 0.975)), :auto)

hr = h2./h1

df3 = DataFrame(hcat(vcat(grid, breaks_extrap), median(hr, dims = 2), quantile.(eachrow(hr), 0.025),  quantile.(eachrow(hr), 0.975)), :auto)

CSV.write(datadir("SOLVDSmps","Base.csv"), df1)
CSV.write(datadir("SOLVDSmps","Trt.csv"), df2)
CSV.write(datadir("SOLVDSmps","HR.csv"), df3)

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
    geom_vline(xintercept = 5.3) +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid",  "dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,NA) + xlim(0,5.2)
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
    geom_vline(xintercept = 5.3) +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid", "dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,NA) + xlim(0,15)

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
    geom_vline(xintercept = 5.3) +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid", "dotdash"))+ ylab("S(t)") + xlab("Time (years)") + ylim(0,1) + xlim(0,5.2)
p3 <- p3 + geom_step(data = km_dat1, aes(x = time, y = surv), col = cbPalette[7], linetype = "dashed") + geom_step(data = km_dat2, aes(x = time, y = surv), col = cbPalette[6], linetype = "dashed")
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
    geom_vline(xintercept = 5.3, linetype = "dotted") +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid", "dotdash")) + ylab("S(t)") + xlab("Time (years)") + ylim(0,1) + xlim(0,15)
p4 <- p4 + geom_step(data = km_dat1, aes(x = time, y = surv), col = cbPalette[7], linetype = "dashed") + geom_step(data = km_dat2, aes(x = time, y = surv), col = cbPalette[6], linetype = "dashed")

p5 <- dat3 %>%
    pivot_longer(Median1:UCI1) %>%
    ggplot(aes(x = Time, y = value, linetype = name)) + geom_step(col = cbPalette[7]) + 
    theme_classic() +
    geom_vline(xintercept = 5.3, linetype = "dotted") +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid", "dotdash")) + ylab("Hazard ratio") + xlab("Time (years)") + ylim(0,2.5) + xlim(0,15)
p5
plot_grid(p1,p2,p4, p5)
#p3
#ggsave($plotsdir("SOLVD_Trt.pdf"), width = 8, height = 4)
"""

### Hosp. -> Death

Random.seed!(9102)
df_all = CSV.read(datadir("SOLVD","SOLVD.csv"), DataFrame)
df = filter(idx -> idx.TRIAL == "P" && idx.EPYTIME > 0.0 && idx.EPY == 1, df_all)
df.CTIME = df.FUTIME - df.EPYTIME
df = filter(idx -> idx.CTIME > 0.0, df)
df.EPC = abs.(ismissing.(df.DDATE) .- 1.0)
sum(df.EPC)
y = df.CTIME/365
maximum(y)
minimum(y)
n = length(y)
breaks = vcat(0.001, collect(0.02:0.1:5.12))
p = 1
cens = df.EPC
covar = fill(1.0, 1, n)
trt = (df.DRUG .== "P").*1.0
covar = [covar; transpose(trt)]
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
x0[2,:] = vcat(x0[2,1], zeros(size(breaks) .-1))
v0[2,:] = vcat(v0[2,1], 1.0, zeros(size(breaks) .-2))
s0[2,:] = vcat(s0[2,1], true, zeros(Int,size(breaks) .-2))

R"""
km = survfit(Surv($y,$cens) ~ $trt)
time1 = km$time[1:km$strata[1]]
time2 = km$time[(km$strata[1] + 1):length(km$time)]
surv1 = km$surv[1:km$strata[1]]
surv2 = km$surv[(km$strata[1] + 1):length(km$surv)]
km_dat1 = data.frame(time = time1, surv = surv1, model = "Placebo")
km_dat2 = data.frame(time = time2, surv = surv2, model = "Enalapril")
km_dat = rbind(km_dat1, km_dat2)
p <- ggplot(data = km_dat, aes(x = time, y = surv, col = model)) + 
    geom_step() + ylim(0, NA) + 
    theme_classic() + scale_colour_manual(values = cbPalette[6:7])
    #geom_step(data = km_dat1, aes(x = time, y = surv), col = cbPalette[7]) + 
    #geom_step(data = km_dat2, aes(x = time, y = surv), col = cbPalette[6]) + ylim(0,NA)
"""

v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 1_000_000
nsmp = 20_000
settings = Settings(nits, nsmp, 1_000_000, 1.0,0.1, 0.5, false, true)

priors1 = BasicPrior(1.0, PC([0.2, 0.2], [2, 2], [0.5, 0.5], Inf), Beta([0.4, 0.4], [10.0, 10.0], [10.0, 10.0]), 1.0, Cts(15.0, 250.0, 5.2), [GompertzBaseline(0.5), GaussLangevin(0.0,0.5)])
Random.seed!(9102)
@time out1 = pem_sample(state0, dat, priors1, settings)


Random.seed!(1237)
grid = sort(unique(out1["Smp_s_loc"][cumsum(out1["Smp_s"],dims = 1)[2,:,:] .> 0.0]))
grid = grid[1:10:length(grid)]

breaks_extrap = collect(5.2:0.02:15)
extrap1 = barker_extrapolation(out1, priors1.diff[1], priors1.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 1)
extrap2 = barker_extrapolation(out1, priors1.diff[2], priors1.grid, breaks_extrap[begin], breaks_extrap[end] + 0.1, breaks_extrap, 2)

extrap2[1,:]
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
    geom_vline(xintercept = 5.3) +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid",  "dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,2) + xlim(0,5.2)
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
    geom_vline(xintercept = 5.3) +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid", "dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,2) + xlim(0,15)

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
    geom_vline(xintercept = 5.3) +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid", "dotdash"))+ ylab("S(t)") + xlab("Time (years)") + ylim(0,1) + xlim(0,5.2)
p3 <- p3 + geom_step(data = km_dat1, aes(x = time, y = surv), col = cbPalette[7], linetype = "dashed") + geom_step(data = km_dat2, aes(x = time, y = surv), col = cbPalette[6], linetype = "dashed")
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
    geom_vline(xintercept = 5.3, linetype = "dotted") +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid", "dotdash")) + ylab("S(t)") + xlab("Time (years)") + ylim(0,1) + xlim(0,15)
p4 <- p4 + geom_step(data = km_dat1, aes(x = time, y = surv), col = cbPalette[7], linetype = "dashed") + geom_step(data = km_dat2, aes(x = time, y = surv), col = cbPalette[6], linetype = "dashed")

p5 <- dat3 %>%
    pivot_longer(Median1:UCI1) %>%
    ggplot(aes(x = Time, y = value, linetype = name)) + geom_step(col = cbPalette[7]) + 
    theme_classic() +
    geom_vline(xintercept = 5.3, linetype = "dotted") + 
    geom_hline(yintercept = 1, linetype = "dashed") + 
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid", "dotdash")) + ylab("Hazard ratio") + xlab("Time (years)") + ylim(0,2.5) + xlim(0,15)
p5
plot_grid(p1,p2,p4, p5)
#p3
#ggsave($plotsdir("SOLVD_Trt.pdf"), width = 8, height = 4)
"""


#### Healthy -> Death


Random.seed!(9102)
df_all = CSV.read(datadir("SOLVD","SOLVD.csv"), DataFrame)
df = filter(idx -> idx.TRIAL == "P" && idx.EPYTIME > 0.0, df_all)
y = df.EPYTIME/365
maximum(y)
minimum(y)
n = length(y)
breaks = vcat(0.001, collect(0.02:0.1:5.22))
p = 1
cens = df.EPX .- df.EPY
sum(cens)
covar = fill(1.0, 1, n)
trt = (df.DRUG .== "P").*1.0
covar = [covar; transpose(trt)]
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
x0[2,:] = vcat(x0[2,1], zeros(size(breaks) .-1))
v0[2,:] = vcat(v0[2,1], 1.0, zeros(size(breaks) .-2))
s0[2,:] = vcat(s0[2,1], true, zeros(Int,size(breaks) .-2))

R"""
km = survfit(Surv($y,$cens) ~ $trt)
time1 = km$time[1:km$strata[1]]
time2 = km$time[(km$strata[1] + 1):length(km$time)]
surv1 = km$surv[1:km$strata[1]]
surv2 = km$surv[(km$strata[1] + 1):length(km$surv)]
km_dat1 = data.frame(time = time1, surv = surv1, model = "Placebo")
km_dat2 = data.frame(time = time2, surv = surv2, model = "Enalapril")
km_dat = rbind(km_dat1, km_dat2)
p <- ggplot(data = km_dat, aes(x = time, y = surv, col = model)) + 
    geom_step() + ylim(0, NA) + 
    theme_classic() + scale_colour_manual(values = cbPalette[6:7])
    #geom_step(data = km_dat1, aes(x = time, y = surv), col = cbPalette[7]) + 
    #geom_step(data = km_dat2, aes(x = time, y = surv), col = cbPalette[6]) + ylim(0,NA)
"""

v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 1_000_000
nsmp = 20_000
settings = Settings(nits, nsmp, 1_000_000, 1.0,0.1, 0.5, false, true)

priors1 = BasicPrior(1.0, PC([0.2, 0.2], [2, 2], [0.5, 0.5], Inf), Beta([0.4, 0.4], [10.0, 10.0], [10.0, 10.0]), 1.0, Cts(15.0, 250.0, 5.22), [GompertzBaseline(0.5), GaussLangevin(0.0,0.5)])
Random.seed!(9102)
@time out1 = pem_sample(state0, dat, priors1, settings)

Random.seed!(1237)
grid = sort(unique(out1["Smp_s_loc"][cumsum(out1["Smp_s"],dims = 1)[2,:,:] .> 0.0]))
grid = grid[1:10:length(grid)]

breaks_extrap = collect(5.2:0.02:15)
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
    geom_vline(xintercept = 5.3) +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid",  "dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,NA) + xlim(0,5.2)
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
    geom_vline(xintercept = 5.3) +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid", "dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,NA) + xlim(0,15)

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
    geom_vline(xintercept = 5.3) +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid", "dotdash"))+ ylab("S(t)") + xlab("Time (years)") + ylim(0,1) + xlim(0,5.2)
p3 <- p3 + geom_step(data = km_dat1, aes(x = time, y = surv), col = cbPalette[7], linetype = "dashed") + geom_step(data = km_dat2, aes(x = time, y = surv), col = cbPalette[6], linetype = "dashed")
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
    geom_vline(xintercept = 5.3, linetype = "dotted") +
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid", "dotdash")) + ylab("S(t)") + xlab("Time (years)") + ylim(0,1) + xlim(0,15)
p4 <- p4 + geom_step(data = km_dat1, aes(x = time, y = surv), col = cbPalette[7], linetype = "dashed") + geom_step(data = km_dat2, aes(x = time, y = surv), col = cbPalette[6], linetype = "dashed")

p5 <- dat3 %>%
    pivot_longer(Median1:UCI1) %>%
    ggplot(aes(x = Time, y = value, linetype = name)) + geom_step(col = cbPalette[7]) + 
    theme_classic() +
    geom_vline(xintercept = 5.3, linetype = "dotted") + 
    geom_hline(yintercept = 1, linetype = "dashed") + 
    theme(legend.position = "none",text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid", "dotdash")) + ylab("Hazard ratio") + xlab("Time (years)") + ylim(0,2.5) + xlim(0,15)
p5
plot_grid(p1,p2, p3, p4, p5, nrow = 3)
#p3
#ggsave($plotsdir("SOLVD_Trt.pdf"), width = 8, height = 4)
"""

# All healthy to death


Random.seed!(9102)
df_all = CSV.read(datadir("SOLVD","SOLVD.csv"), DataFrame)
df = filter(idx -> idx.TRIAL == "P" && idx.EPYTIME > 0.0, df_all)
y = df.FUTIME/365
maximum(y)
minimum(y)
n = length(y)
breaks = vcat(0.001, collect(0.02:0.1:5.22))
p = 1
cens = abs.(ismissing.(df.DDATE) .- 1.0)
sum(cens)
covar = fill(1.0, 1, n)
trt = (df.DRUG .== "P").*1.0
covar = [covar; transpose(trt)]
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
x0[2,:] = vcat(x0[2,1], zeros(size(breaks) .-1))
v0[2,:] = vcat(v0[2,1], 1.0, zeros(size(breaks) .-2))
s0[2,:] = vcat(s0[2,1], true, zeros(Int,size(breaks) .-2))

R"""
km = survfit(Surv($y,$cens) ~ $trt)
time1 = km$time[1:km$strata[1]]
time2 = km$time[(km$strata[1] + 1):length(km$time)]
surv1 = km$surv[1:km$strata[1]]
surv2 = km$surv[(km$strata[1] + 1):length(km$surv)]
km_dat1 = data.frame(time = time1, surv = surv1, model = "Placebo")
km_dat2 = data.frame(time = time2, surv = surv2, model = "Enalapril")
km_dat = rbind(km_dat1, km_dat2)
p <- ggplot(data = km_dat, aes(x = time, y = surv, col = model)) + 
    geom_step() + ylim(0, 1) + 
    theme_classic() + scale_colour_manual(values = cbPalette[6:7])
    #geom_step(data = km_dat1, aes(x = time, y = surv), col = cbPalette[7]) + 
    #geom_step(data = km_dat2, aes(x = time, y = surv), col = cbPalette[6]) + ylim(0,NA)
"""

v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0))
nits = 1_000_000
nsmp = 20_000
settings = Settings(nits, nsmp, 1_000_000, 5.0,5.0, 5.0, false, true)

priors1 = BasicPrior(1.0, PC([0.2, 0.2], [0.1, 0.1], [0.5, 0.5], Inf), Beta([0.5, 0.25], [10.0, 10.0], [5.0, 15.0]), 1.0, Cts(15.0, 250.0, 5.22), [GompertzBaseline(0.5), GaussLangevin(0.0,0.5)])
Random.seed!(9102)
@time out1 = pem_sample(state0, dat, priors1, settings)


#priors2 = BasicPrior(1.0, InvGamma([0.2,0.2], [0.5,0.5], [0.5,0.5]), Beta([0.4, 0.4], [10.0, 10.0], [10.0, 10.0]), 1.0, Cts(15.0, 250.0, 5.22), [GompertzBaseline(0.5), GaussLangevin(0.0,0.5)])
#Random.seed!(9102)
#@time out2 = pem_sample(state0, dat, priors2, settings)


Random.seed!(1237)
grid = sort(unique(out1["Smp_s_loc"][cumsum(out1["Smp_s"],dims = 1)[2,:,:] .> 0.0]))
grid = grid[1:10:length(grid)]

breaks_extrap = collect(5.2:0.02:15)
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

CSV.write(datadir("SOLVDSmps","ACM_Base.csv"), df1)
CSV.write(datadir("SOLVDSmps","ACM_Trt.csv"), df2)
CSV.write(datadir("SOLVDSmps","ACM_HR.csv"), df3)

plot(out1["Smp_J"])
plot(cumsum(out1["Smp_s"] .== 1, dims = 2)[1,end,:])
plot(cumsum(out1["Smp_s"] .== 1, dims = 2)[2,end,:])

plot(vec(log.(h1[5,:])))
plot(vec(log.(h1[1_000,:])))
plot(vec(log.(h1[2_000,:])))
plot(vec(log.(h1[3_000,:])))
plot(vec(log.(h1[4_000,:])))
plot(vec(log.(h1[5_000,:])))
plot(vec(log.(h1[6_500,:])))

plot(vec(log.(h2[5,:])))
plot(vec(log.(h2[1_000,2_000:3_000])))
plot(vec(log.(h2[2_000,:])))
plot(vec(log.(h2[3_000,:])))
plot(vec(log.(h2[4_000,:])))
plot(vec(log.(h2[5_000,:])))
plot(vec(log.(h2[6_000,:])))
plot(vec(log.(h2[6_500,:])))


plot(vec(out1["Smp_σ"][1,:]))
plot(vec(out1["Smp_σ"][2,:]))
plot(scatter(log.(vec(out1["Smp_σ"][1,:])),log.(vec(out1["Smp_σ"][2,:]))))
plot(vec(out2["Smp_σ"][1,:]))
plot(vec(out2["Smp_σ"][2,:]))
plot(scatter(log.(vec(out2["Smp_σ"][1,:])),log.(vec(out2["Smp_σ"][2,:]))))

plot(vec(out1["Smp_ω"][1,:]))
plot(vec(out1["Smp_ω"][2,:]))
plot(vec(out2["Smp_ω"][1,:]))
plot(vec(out2["Smp_ω"][2,:]))

df1 = CSV.read(datadir("SOLVDSmps","ACM_Base.csv"), DataFrame)
df2 = CSV.read(datadir("SOLVDSmps","ACM_Trt.csv"), DataFrame)
df3 = CSV.read(datadir("SOLVDSmps","ACM_HR.csv"), DataFrame)

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
    geom_vline(xintercept = 5.3) +
    theme(text = element_text(size = 10)) + scale_colour_manual(values = cbPalette[c(6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid",  "dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.2) + xlim(0,5.2)
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
    geom_vline(xintercept = 5.3) +
    theme(legend.position = "none",text = element_text(size = 10)) + scale_colour_manual(values = cbPalette[c(6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid", "dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.5) + xlim(0,15)

km = survfit(Surv($y,$cens) ~ $trt)
time1 = km$time[1:km$strata[1]]
time2 = km$time[(km$strata[1] + 1):length(km$time)]
surv1 = km$surv[1:km$strata[1]]
surv2 = km$surv[(km$strata[1] + 1):length(km$surv)]
km_dat1 = data.frame(time = time1, surv = surv1)
km_dat2 = data.frame(time = time2, surv = surv2)

p3 <- dat2 %>%
    pivot_longer(c(Median1,Median2)) %>%
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
    subset(Time < 5.3) %>%
    ggplot(aes(x = Time, y = value)) + geom_step(aes(col = Arm, linetype = Stat)) + 
    theme_classic() + 
    geom_vline(xintercept = 5.3) +
    theme(legend.position = "none",text = element_text(size = 10)) + scale_colour_manual(values = cbPalette[c(6,7)]) +
    scale_linetype_manual(values = c("solid","solid"))+ ylab("S(t)") + xlab("Time (years)") + xlim(0,5.2)
p3 <- p3 + geom_step(data = km_dat1, aes(x = time, y = surv), col = cbPalette[7], linetype = "dashed") + geom_step(data = km_dat2, aes(x = time, y = surv), col = cbPalette[6], linetype = "dashed")
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
    geom_vline(xintercept = 5.3, linetype = "dotted") +
    theme(legend.position = "none",text = element_text(size = 10)) + scale_colour_manual(values = cbPalette[c(6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid", "dotdash")) + ylab("S(t)") + xlab("Time (years)") + ylim(0,1) + xlim(0,15)
#p4 <- p4 + geom_step(data = km_dat1, aes(x = time, y = surv), col = cbPalette[7], linetype = "dashed") + geom_step(data = km_dat2, aes(x = time, y = surv), col = cbPalette[6], linetype = "dashed")

p5 <- dat3 %>%
    pivot_longer(Median1:UCI1) %>%
    ggplot(aes(x = Time, y = log(value), linetype = name)) + geom_step(col = cbPalette[7]) + 
    theme_classic() +
    geom_vline(xintercept = 5.3, linetype = "dotted") + 
    geom_hline(yintercept = 1, linetype = "dashed") + 
    theme(legend.position = "none",text = element_text(size = 10)) + scale_colour_manual(values = cbPalette[c(6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid", "dotdash")) + ylab("Hazard ratio") + xlab("Time (years)") #+ ylim(0,2.5) + xlim(0,15)
p5
plot_grid(p1,p2, p3, p4, nrow = 2)
#p5
#ggsave($plotsdir("SOLVD_ACM.pdf"), width = 8, height = 6)
"""