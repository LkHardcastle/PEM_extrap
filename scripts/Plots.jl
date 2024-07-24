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

## Extrapolation figure
t = collect(0.3:0.3:3.9)
ht = [0.3,0.9,0.6,0.5,0.3,0.1,0.1,0.1,0.15,0.16,0.2,0.25,0.3]

df = DataFrame(hcat(t, ht, vcat(fill(1,8),fill(2,5))), :auto)

R"""
$df %>%
    subset(x1 < 2.71) %>%
    ggplot(aes(x = x1, y = x2)) + geom_smooth(se = F, col = cbPalette[7]) + theme_classic() +
    theme(text = element_text(size = 20)) +
    xlab("Time (years)") + ylab("h(t)") + ylim(0,0.7) + geom_vline(xintercept = 2.7, linetype = "dashed") + 
    annotate("segment", x = 2.7, y = 0.15, xend = 4, yend = 0.4, col = cbPalette[6], linetype = "dashed") + 
    annotate("segment", x = 2.7, y = 0.15, xend = 4, yend = 0.12, col = cbPalette[4], linetype = "dashed") + 
    annotate("segment", x = 2.7, y = 0.15, xend = 4, yend = 0.7, col = cbPalette[7], linetype = "dashed")
    ggsave($plotsdir("Extrap.png"), width = 9, height = 4)
    ggsave($plotsdir("Extrap.pdf"), width = 14, height = 6)
"""

R"""
$df %>%
    subset(x1 < 2.71) %>%
    ggplot(aes(x = x1, y = x2)) + geom_smooth(se = F, col = cbPalette[7], linetype = "solid") + theme_classic() +
    theme(text = element_text(size = 20)) +
    xlab("Time (years)") + ylab("h(t)") + ylim(0,0.7) + geom_vline(xintercept = 2.7, linetype = "dashed") + 
    annotate("segment", x = 0, y = 0.4, xend = 0.5, yend = 0.4, col = cbPalette[4], linetype = "solid") + 
    annotate("segment", x = 0.5, y = 0.5, xend = 1, yend = 0.5, col = cbPalette[4], linetype = "solid") + 
    annotate("segment", x = 1, y = 0.45, xend = 1.5, yend = 0.45, col = cbPalette[4], linetype = "solid") + 
    annotate("segment", x = 1.5, y = 0.2, xend = 2, yend = 0.2, col = cbPalette[4], linetype = "solid") + 
    annotate("segment", x = 2, y = 0.1, xend = 2.5, yend = 0.1, col = cbPalette[4], linetype = "solid") + 
    annotate("segment", x = 2.5, y = 0.15, xend = 3, yend = 0.15, col = cbPalette[4], linetype = "solid") + 
    annotate("segment", x = 3, y = 0.15, xend = 3.5, yend = 0.15, col = cbPalette[4], linetype = "solid") + 
    annotate("segment", x = 3.5, y = 0.15, xend = 4, yend = 0.15, col = cbPalette[4], linetype = "solid") +
    annotate("segment", x = 2.5, y = 0.22, xend = 3, yend = 0.22, col = cbPalette[4], linetype = "dashed") +
    annotate("segment", x = 3, y = 0.27, xend = 3.5, yend = 0.27, col = cbPalette[4], linetype = "dashed") + 
    annotate("segment", x = 3.5, y = 0.33, xend = 4, yend = 0.33, col = cbPalette[4], linetype = "dashed") +
    annotate("segment", x = 2.5, y = 0.08, xend = 3, yend = 0.08, col = cbPalette[4], linetype = "dashed") +
    annotate("segment", x = 3, y = 0.05, xend = 3.5, yend = 0.05, col = cbPalette[4], linetype = "dashed") + 
    annotate("segment", x = 3.5, y = 0.02, xend = 4, yend = 0.02, col = cbPalette[4], linetype = "dashed")
    ggsave($plotsdir("PWEXP.pdf"), width = 14, height = 6)
"""


Random.seed!(12515)
n = 0
y = rand(Exponential(1.0),n)
breaks = collect(0.5:0.5:1.0)
p = 1
cens = fill(1.0,n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0

Random.seed!(3463)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
state0 = ECMC2(x0, v0, s0, t0, true, findall(s0))
nits = 20
nsmp = 10000
settings = Settings(nits, nsmp, 1_000_000, 0.01,0.0, 0.1, false, true)
priors = BasicPrior(1.0, FixedV(1.0), FixedW(0.5), 1.0)
@time out3 = pem_sample(state0, dat, priors, settings)

plot(vec(out3["Smp_t"]), vec(out3["Smp_trans"][1,1,:]))
plot!(vec(out3["Smp_t"]), vec(out3["Smp_trans"][1,2,:]))

plot(vec(out3["Smp_t"]), vec(out3["Smp_x"][1,1,:]))
plot!(vec(out3["Smp_t"]), vec(out3["Smp_x"][1,2,:]))

t = vec(out3["Sk_t"])
#x1 = vec(out3["Smp_trans"][1,1,:])
#x2 = vec(out3["Smp_trans"][1,2,:])
y1 = vec(out3["Sk_x"][1,1,:])
y2 = vec(out3["Sk_x"][1,2,:])
x1 = copy(y1)
x2 = y1 + y2
df = DataFrame([t, x1, x2, y1, y2], [:t, :x1, :x2, :y1, :y2])

R"""
p1 <- $df %>%
    pivot_longer(x2:x1) %>%
    ggplot(aes(x = t, y = value, col = factor(name, levels = c("x2","x1")))) + geom_line() +
    theme_classic() + scale_colour_manual(values = cbPalette[7:6]) + ylab("State") + xlab("Sampler time (arbitrary units)") +
    theme(legend.position = "none", text = element_text(size = 20))
p2 <- $df %>%
    pivot_longer(y1:y2) %>%
    ggplot(aes(x = t, y = value, col = name)) + geom_line() +
    theme_classic() + scale_colour_manual(values = cbPalette[6:7]) + ylab("State") + xlab("Sampler time (arbitrary units)") +
    theme(legend.position = "none", text = element_text(size = 20))
plot_grid(p1,p2)
ggsave($plotsdir("SM.pdf"), width = 14, height = 6)
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
t0 = 0.0
priors = BasicPrior(1.0, FixedV(1.0), FixedW(0.5), 1.0)
state0 = BPS(x0, v0, s0, t0, findall(s0))
nits = 100_000
nsmp = 100000
Random.seed!(123)
settings = Settings(nits, nsmp, 100000, 0.5,0.0, 1.0, false, true)
@time out1 = pem_sample(state0, dat, priors, settings)
settings = Settings(nits, nsmp, 100000, 0.5,0.0, 0.2, false, true)
@time out2 = pem_sample(state0, dat, priors, settings)
settings = Settings(nits, nsmp, 100000, 0.5,0.0, 0.01, false, true)
@time out3 = pem_sample(state0, dat, priors, settings)
state0 = ECMC2(x0, v0, s0, t0, true, findall(s0))
settings = Settings(nits, nsmp, 100000, 0.5,0.0, 1.0, false, true)
@time out4 = pem_sample(state0, dat, priors, settings)
settings = Settings(nits, nsmp, 100000, 0.5,0.0, 0.2, false, true)
@time out5 = pem_sample(state0, dat, priors, settings)
settings = Settings(nits, nsmp, 100000, 0.5,0.0, 0.01, false, true)
@time out6 = pem_sample(state0, dat, priors, settings)

smps1 = out1["Smp_trans"]
s1 = view(smps1, 1, :, :)
plot(vcat(0,breaks), vcat(mean(exp.(s1), dims = 2), mean(exp.(s1), dims = 2)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.025),quantile.(eachrow(exp.(s1)), 0.025)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.975),quantile.(eachrow(exp.(s1)), 0.975)[end]),linetype=:steppost, ylim = (0,1))

smps1 = out2["Smp_trans"]
s1 = view(smps1, 1, :, :)
plot(vcat(0,breaks), vcat(mean(exp.(s1), dims = 2), mean(exp.(s1), dims = 2)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.025),quantile.(eachrow(exp.(s1)), 0.025)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.975),quantile.(eachrow(exp.(s1)), 0.975)[end]),linetype=:steppost, ylim = (0,1))

smps1 = out3["Smp_trans"]
s1 = view(smps1, 1, :, :)
plot(vcat(0,breaks), vcat(mean(exp.(s1), dims = 2), mean(exp.(s1), dims = 2)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.025),quantile.(eachrow(exp.(s1)), 0.025)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.975),quantile.(eachrow(exp.(s1)), 0.975)[end]),linetype=:steppost, ylim = (0,1))

smps1 = out4["Smp_trans"]
s1 = view(smps1, 1, :, :)
plot(vcat(0,breaks), vcat(mean(exp.(s1), dims = 2), mean(exp.(s1), dims = 2)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.025),quantile.(eachrow(exp.(s1)), 0.025)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.975),quantile.(eachrow(exp.(s1)), 0.975)[end]),linetype=:steppost, ylim = (0,1))

smps1 = out5["Smp_trans"]
s1 = view(smps1, 1, :, :)
plot(vcat(0,breaks), vcat(mean(exp.(s1), dims = 2), mean(exp.(s1), dims = 2)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.025),quantile.(eachrow(exp.(s1)), 0.025)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.975),quantile.(eachrow(exp.(s1)), 0.975)[end]),linetype=:steppost, ylim = (0,1))

smps1 = out6["Smp_trans"]
s1 = view(smps1, 1, :, :)
plot(vcat(0,breaks), vcat(mean(exp.(s1), dims = 2), mean(exp.(s1), dims = 2)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.025),quantile.(eachrow(exp.(s1)), 0.025)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.975),quantile.(eachrow(exp.(s1)), 0.975)[end]),linetype=:steppost, ylim = (0,1))


s1 = view(exp.(out1["Smp_trans"]), 1, :, :)
d1 = DataFrame(hcat(breaks, mean(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.975), fill(1, 31), fill(1, 31)), :auto)
s2 = view(exp.(out2["Smp_trans"]), 1, :, :)
d2 = DataFrame(hcat(breaks, mean(s2, dims = 2), quantile.(eachrow(s2), 0.025), quantile.(eachrow(s2), 0.975), fill(0.2, 31), fill(1, 31)), :auto)
s3 = view(exp.(out3["Smp_trans"]), 1, :, :)
d3 = DataFrame(hcat(breaks, mean(s3, dims = 2), quantile.(eachrow(s3), 0.025), quantile.(eachrow(s3), 0.975), fill(0.01, 31), fill(1, 31)), :auto)
s4 = view(exp.(out4["Smp_trans"]), 1, :, :)
d4 = DataFrame(hcat(breaks, mean(s4, dims = 2), quantile.(eachrow(s4), 0.025), quantile.(eachrow(s4), 0.975), fill(1, 31), fill(2, 31)), :auto)
s5 = view(exp.(out5["Smp_trans"]), 1, :, :)
d5 = DataFrame(hcat(breaks, mean(s5, dims = 2), quantile.(eachrow(s5), 0.025), quantile.(eachrow(s5), 0.975), fill(0.2, 31), fill(2, 31)), :auto)
s6 = view(exp.(out6["Smp_trans"]), 1, :, :)
d6 = DataFrame(hcat(breaks, mean(s6, dims = 2), quantile.(eachrow(s6), 0.025), quantile.(eachrow(s6), 0.975), fill(0.01, 31), fill(2, 31)), :auto)

df = vcat(d1,d2,d3,d4,d5,d6)
R"""
$df
"""
R"""
dat = data.frame($df)
colnames(dat) <- c("Time","Mean","LCI","UCI", "Ref","Method") 
dat
"""
R"""
dat %>%
    mutate(Method = ifelse(Method == 1, "BPS", "FECMC"),
            Ref = ifelse(Ref == 1, "Ref = 1",
            ifelse(Ref == 0.2, "Ref = 0.2", "Ref = 0.01"))) %>%
    pivot_longer(Mean:UCI) %>%
    ggplot(aes(x = Time, y = value, col = name)) + geom_step() +
    theme_classic() + ylab("h(t)") + xlab("Time (years)") + ylim(0,1) + 
    theme(legend.position = "none",
        panel.background = element_rect(fill = NA, colour = "black")) +
    scale_colour_manual(values = cbPalette[c(7,6,7)]) + facet_grid(rows = vars(Method), cols = vars(Ref))
    ggsave($plotsdir("Ref.pdf"), width = 14, height = 6)
"""

Random.seed!(123)
df = CSV.read(datadir("colon.csv"), DataFrame)
y = df.years
maximum(y)
n = length(y)
breaks = collect(0.1:0.1:5)
p = 1
cens = df.status
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, t0, true, findall(s0))
nits = 100_000
nsmp = 100000
settings = Settings(nits, nsmp, 100000, 0.5,1.0, 0.02, false, true)
Random.seed!(123)
priors = BasicPrior(1.0, FixedV(0.5), Beta(0.4, 10.0, 10.0), 1.0)
@time out1 = pem_sample(state0, dat, priors, settings)

s1 = view(exp.(out1["Smp_trans"]), 1, :, :)
df = DataFrame(hcat(breaks, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
R"""
dat = data.frame($df)
colnames(dat) <- c("Time","Mean","LCI","Q1","Q4","UCI") 
dat
"""
R"""
dat %>%
    pivot_longer(Mean:UCI) %>%
    ggplot(aes(x = Time, y = value, col = name, linetype = name)) + geom_step() +
    theme_classic() + 
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,4,4,6)]) +
    scale_linetype_manual(values = c("dotdash","solid","dashed","dashed","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,1) +
    geom_vline(xintercept = 3, linetype = "dashed")
    ggsave($plotsdir("Example.pdf"), width = 14, height = 6)
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
t0 = 0.0
state0 = ECMC2(x0, v0, s0, t0, true, findall(s0))
nits = 100_000
nsmp = 100000
settings = Settings(nits, nsmp, 100000, 0.5,1.0, 0.02, false, true)
Random.seed!(123)
priors = BasicPrior(1.0, PC(0.2, 2, 0.5, 1, Inf), Beta(0.4, 10.0, 10.0), 1.0)
@time out1 = pem_sample(state0, dat, priors, settings)
priors = BasicPrior(1.0, PC(0.2, 2, 0.5, 1, Inf), Beta(0.4, 10.0, 10.0), 0.0)
@time out2 = pem_sample(state0, dat, priors, settings)

s1 = view(exp.(out1["Smp_trans"]), 1, :, :)
df1 = DataFrame(hcat(breaks, mean(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
s2 = view(exp.(out2["Smp_trans"]), 1, :, :)
df2 = DataFrame(hcat(breaks, mean(s2, dims = 2), quantile.(eachrow(s2), 0.025), quantile.(eachrow(s2), 0.25), quantile.(eachrow(s2), 0.75), quantile.(eachrow(s2), 0.975)), :auto)
R"""
dat1 = data.frame($df1)
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI") 
"""
R"""
dat2 = data.frame($df2)
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI") 
"""

R"""
p1 <- dat1 %>%
    pivot_longer(Mean:UCI) %>%
    ggplot(aes(x = Time, y = value, col = name, linetype = name)) + geom_step() +
    theme_classic() + 
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,4,4,6)]) +
    scale_linetype_manual(values = c("dotdash","solid","dashed","dashed","dotdash")) + ylab("h(t)") + xlab("Time (years)")
"""
R"""    
p2 <- dat2 %>%
    pivot_longer(Mean:UCI) %>%
    ggplot(aes(x = Time, y = value, col = name, linetype = name)) + geom_step() +
    theme_classic() + 
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6,7,4,4,6)]) +
    scale_linetype_manual(values = c("dotdash","solid","dashed","dashed","dotdash")) + ylab("h(t)") + xlab("Time (years)")
p2
#plot_grid(p1,p2)
#ggsave($plotsdir("SnS.pdf"), width = 14, height = 6)
"""
