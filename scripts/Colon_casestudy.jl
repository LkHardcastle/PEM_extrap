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
nits = 50000
nsmp = 100000
settings = Settings(nits, nsmp, 1_000_000, 2.0,0.5, 0.5, false, true)

priors1 = BasicPrior(1.0, FixedV(0.5), FixedW(0.5), 0.0, Fixed(), RandomWalk())
priors2 = BasicPrior(1.0, FixedV(0.5), FixedW(0.5), 0.0, Fixed(), GaussLangevin(0.0,1.0))
priors3 = BasicPrior(1.0, FixedV(0.5), FixedW(0.5), 0.0, Fixed(), GammaLangevin(1.0,1.0))
Random.seed!(9102)
@time out1 = pem_sample(state0, dat, priors1, settings)
@time out2 = pem_sample(state0, dat, priors2, settings)
@time out3 = pem_sample(state0, dat, priors3, settings)

priors4 = BasicPrior(1.0, FixedV(0.5), FixedW(0.5), 1.0, Fixed(), RandomWalk())
priors5 = BasicPrior(1.0, FixedV(0.5), FixedW(0.5), 1.0, Fixed(), GaussLangevin(0.0,1.0))
priors6 = BasicPrior(1.0, FixedV(0.5), FixedW(0.5), 1.0, Fixed(), GammaLangevin(1.0,1.0))
Random.seed!(9102)
@time out4 = pem_sample(state0, dat, priors4, settings)
@time out5 = pem_sample(state0, dat, priors5, settings)
@time out6 = pem_sample(state0, dat, priors6, settings)

priors7 = BasicPrior(1.0, PC(0.2, 2, 0.5, 1, Inf), Beta(0.4, 10.0, 10.0), 1.0, Fixed(), RandomWalk())
priors8 = BasicPrior(1.0, PC(0.2, 2, 0.5, 1, Inf), Beta(0.4, 10.0, 10.0), 1.0, Fixed(), GaussLangevin(0.0,1.0))
priors9 = BasicPrior(1.0, PC(0.2, 2, 0.5, 1, Inf), Beta(0.4, 10.0, 10.0), 1.0, Fixed(), GammaLangevin(1.0,1.0))
Random.seed!(9102)
@time out7 = pem_sample(state0, dat, priors7, settings)
@time out8 = pem_sample(state0, dat, priors8, settings)
@time out9 = pem_sample(state0, dat, priors9, settings)

priors10 = BasicPrior(1.0, FixedV(0.5), FixedW(0.5), 1.0, Cts(10.0, 50.0, 3.5), RandomWalk())
priors11 = BasicPrior(1.0, FixedV(0.5), FixedW(0.5), 1.0, Cts(10.0, 50.0, 3.5), GaussLangevin(0.0,1.0))
priors12 = BasicPrior(1.0, FixedV(0.5), FixedW(0.5), 1.0, Cts(10.0, 50.0, 3.5), GammaLangevin(1.0,1.0))
Random.seed!(9102)
@time out10 = pem_sample(state0, dat, priors10, settings)
@time out11 = pem_sample(state0, dat, priors11, settings)
@time out12 = pem_sample(state0, dat, priors12, settings)

priors13 = BasicPrior(1.0, PC(0.2, 2, 0.5, 1, Inf), Beta(0.4, 10.0, 10.0), 1.0, Cts(10.0, 50.0, 3.5), RandomWalk())
priors14 = BasicPrior(1.0, PC(0.2, 2, 0.5, 1, Inf), Beta(0.4, 10.0, 10.0), 1.0, Cts(10.0, 50.0, 3.5), GaussLangevin(0.0,1.0))
priors15 = BasicPrior(1.0, PC(0.2, 2, 0.5, 1, Inf), Beta(0.4, 10.0, 10.0), 1.0, Cts(10.0, 50.0, 3.5), GammaLangevin(1.0,1.0))
Random.seed!(9102)
@time out13 = pem_sample(state0, dat, priors13, settings)
@time out14 = pem_sample(state0, dat, priors14, settings)
@time out15 = pem_sample(state0, dat, priors15, settings)


s1 = view(exp.(out1["Smp_trans"]), 1, :, :)
df1 = DataFrame(hcat(breaks, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
s1 = view(exp.(out2["Smp_trans"]), 1, :, :)
df2 = DataFrame(hcat(breaks, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
s1 = view(exp.(out3["Smp_trans"]), 1, :, :)
df3 = DataFrame(hcat(breaks, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
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
#ggsave($plotsdir("CtsColonCovariate.pdf"), width = 8, height = 6)
"""

s1 = view(exp.(out4["Smp_trans"]), 1, :, :)
df1 = DataFrame(hcat(breaks, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
s1 = view(exp.(out5["Smp_trans"]), 1, :, :)
df2 = DataFrame(hcat(breaks, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
s1 = view(exp.(out6["Smp_trans"]), 1, :, :)
df3 = DataFrame(hcat(breaks, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
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
#ggsave($plotsdir("CtsColonCovariate.pdf"), width = 8, height = 6)
"""

s1 = view(exp.(out7["Smp_trans"]), 1, :, :)
df1 = DataFrame(hcat(breaks, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
s1 = view(exp.(out8["Smp_trans"]), 1, :, :)
df2 = DataFrame(hcat(breaks, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
s1 = view(exp.(out9["Smp_trans"]), 1, :, :)
df3 = DataFrame(hcat(breaks, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
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
#ggsave($plotsdir("CtsColonCovariate.pdf"), width = 8, height = 6)
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
#ggsave($plotsdir("CtsColonCovariate.pdf"), width = 8, height = 6)
"""

plot(view(exp.(out10["Smp_trans"]),1,30,2984:2986))
ind = 2984
plot(out10["Smp_s_loc"][findall(out10["Smp_s_loc"][:,ind] .!= Inf),ind], view(out10["Smp_trans"],1,:,ind)[findall(out10["Smp_s_loc"][:,ind] .!= Inf)], seriestype = :steppre)
plot!(out10["Smp_s_loc"][findall(out10["Smp_s_loc"][:,ind] .!= Inf),ind], view(out10["Smp_x"],1,:,ind)[findall(out10["Smp_s_loc"][:,ind] .!= Inf)], seriestype = :steppre)
scatter!(dat.y,zeros(size(dat.y)), markershape = :cross)
view(out10["Smp_trans"],1,1:10,ind)
view(out10["Smp_x"],1,1:10,ind)
view(out10["Smp_s"],1,1:10,ind)
view(cumsum(out10["Smp_x"], dims = 2),1,1:10,ind)



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
#ggsave($plotsdir("CtsColonCovariate.pdf"), width = 8, height = 6)
"""