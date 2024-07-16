using DrWatson
@quickactivate "PEM_extrap"
# For src
using DataStructures, LinearAlgebra, Distributions, Random, Optim, Roots, SpecialFunctions
using Plots, CSV, DataFrames, RCall, Pkg

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

Random.seed!(12515)
n = 0
y = rand(Exponential(1.0),n)
breaks = collect(0.5:0.5:5)
p = 1
cens = fill(1.0,n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = BPS(x0, v0, s0, t0, findall(s0))
priors = BasicPrior(1.0, FixedV(1.0), FixedW(0.5), 0.0)
nits = 10_000
nsmp = 100000
settings = Settings(nits, nsmp, 1_000_000, 0.5,0.0, 0.1, false, true)
Random.seed!(123)
@time out1 = pem_sample(state0, dat, priors, settings)
settings = Settings(nits, nsmp, 100000, 0.5,0.0, 0.5, false, true)
@time out11 = pem_sample(state0, dat, priors, settings)
settings = Settings(nits, nsmp, 100000, 0.5,0.0, 1.0, false, true)
@time out12 = pem_sample(state0, dat, priors, settings)

Random.seed!(34734)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
state0 = ECMC2(x0, v0, s0, t0, true, findall(s0))
nits = 10_000
nsmp = 100000
settings = Settings(nits, nsmp, 1_000_000, 0.5,0.0, 0.1, false, true)
priors = BasicPrior(1.0, FixedV(1.0), FixedW(0.5), 0.0)
@time out3 = pem_sample(state0, dat, priors, settings)
Random.seed!(123)
settings = Settings(nits, nsmp, 1_000_000, 0.5,0.0, 0.5, false, true)
@time out31 = pem_sample(state0, dat, priors, settings)
settings = Settings(nits, nsmp, 1_000_000, 0.5,0.0, 1.0, false, true)
@time out32 = pem_sample(state0, dat, priors, settings)


sk1 = view(out1["Sk_x"], 1, :, 1:100)
sk2 = view(out11["Sk_x"], 1, :, 1:100)
sk3 = view(out12["Sk_x"], 1, :, 1:100)
sk4 = view(out3["Sk_x"], 1, :, 1:100)
sk5 = view(out31["Sk_x"], 1, :, 1:100)
sk6 = view(out32["Sk_x"], 1, :, 1:100)

R"""
p1 <- $sk1 %>%
    t() %>%
    data.frame() %>%
    ggplot(aes(x = X1, y = X2)) + geom_path(col = cbPalette[6]) +
    theme_classic() + xlab("") + ylab("")
p2 <- $sk2 %>%
    t() %>%
    data.frame() %>%
    ggplot(aes(x = X1, y = X2)) + geom_path(col = cbPalette[6]) +
    theme_classic() + xlab("") + ylab("")
p3 <- $sk3 %>%
    t() %>%
    data.frame() %>%
    ggplot(aes(x = X1, y = X2)) + geom_path(col = cbPalette[6]) +
    theme_classic() + xlab("") + ylab("")
p4 <- $sk4 %>%
    t() %>%
    data.frame() %>%
    ggplot(aes(x = X1, y = X2)) + geom_path(col = cbPalette[6]) +
    theme_classic() + xlab("") + ylab("")
p5 <- $sk5 %>%
    t() %>%
    data.frame() %>%
    ggplot(aes(x = X1, y = X2)) + geom_path(col = cbPalette[6]) +
    theme_classic() + xlab("") + ylab("")+ xlab("") + ylab("")
p6 <- $sk6 %>%
    t() %>%
    data.frame() %>%
    ggplot(aes(x = X1, y = X2)) + geom_path(col = cbPalette[6]) +
    theme_classic() + xlab("") + ylab("")
plot_grid(p1,p2,p3,p4,p5,p6)
ggsave($plotsdir("BPS.pdf"), width = 6, height = 4)
"""
plot(vec(sk1[1,:]),vec(sk1[2,:]))
plot(vec(sk2[1,:]),vec(sk2[2,:]))
plot(vec(sk3[1,:]),vec(sk3[2,:]))
plot(vec(sk4[1,:]),vec(sk4[2,:]))
plot(vec(sk5[1,:]),vec(sk5[2,:]))
plot(vec(sk6[1,:]),vec(sk6[2,:]))


sk1 = view(out1["Smp_x"], 1, :, :)
sk2 = view(out11["Smp_x"], 1, :, :)
sk3 = view(out12["Smp_x"], 1, :, :)
sk4 = view(out3["Smp_x"], 1, :, :)
sk5 = view(out31["Smp_x"], 1, :, :)
sk6 = view(out32["Smp_x"], 1, :, :)

mean(eachcol(sk1))
quantile.(eachrow(sk1), 0.025)
quantile.(eachrow(sk1), 0.975)
mean(eachcol(sk2))
quantile.(eachrow(sk2), 0.025)
quantile.(eachrow(sk2), 0.975)
mean(eachcol(sk3))
quantile.(eachrow(sk3), 0.025)
quantile.(eachrow(sk3), 0.975)
mean(eachcol(sk4))
quantile.(eachrow(sk4), 0.025)
quantile.(eachrow(sk4), 0.975)
mean(eachcol(sk5))
quantile.(eachrow(sk5), 0.025)
quantile.(eachrow(sk5), 0.975)
mean(eachcol(sk6))
quantile.(eachrow(sk6), 0.025)
quantile.(eachrow(sk6), 0.975)

Random.seed!(12515)
n = 0
y = rand(Exponential(1.0),n)
breaks = collect(0.5:0.5:5)
p = 1
cens = fill(1.0,n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = BPS(x0, v0, s0, t0, findall(s0))
priors = BasicPrior(1.0, FixedV(1.0), FixedW(0.5), 1.0)
nits = 10_000
nsmp = 100000
settings = Settings(nits, nsmp, 1_000_000, 0.5,0.0, 0.1, false, true)
Random.seed!(123)
@time out1 = pem_sample(state0, dat, priors, settings)
settings = Settings(nits, nsmp, 100000, 0.5,0.0, 0.5, false, true)
@time out11 = pem_sample(state0, dat, priors, settings)
settings = Settings(nits, nsmp, 100000, 0.5,0.0, 1.0, false, true)
@time out12 = pem_sample(state0, dat, priors, settings)

Random.seed!(34734)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
state0 = ECMC2(x0, v0, s0, t0, true, findall(s0))
nits = 10_000
nsmp = 100000
settings = Settings(nits, nsmp, 1_000_000, 0.5,0.0, 0.1, false, true)
priors = BasicPrior(1.0, FixedV(1.0), FixedW(0.5), 1.0)
@time out3 = pem_sample(state0, dat, priors, settings)
Random.seed!(123)
settings = Settings(nits, nsmp, 1_000_000, 0.5,0.0, 0.5, false, true)
@time out31 = pem_sample(state0, dat, priors, settings)
settings = Settings(nits, nsmp, 1_000_000, 0.5,0.0, 1.0, false, true)
@time out32 = pem_sample(state0, dat, priors, settings)

mean(eachcol(sk1.== 0.0))
mean(eachcol(sk2.== 0.0))
mean(eachcol(sk3.== 0.0))
mean(eachcol(sk4.== 0.0))
mean(eachcol(sk5.== 0.0))
mean(eachcol(sk6.== 0.0))

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
    ggsave($plotsdir("Ref.pdf"), width = 6, height = 4)
"""

R"""
dat <- $d1 %>%
    data.frame() 
    colnames(dat) <- c("Time","Mean","LCI","UCI") 
p1 <- dat %>%
    pivot_longer(Mean:UCI) %>%
    ggplot(aes(x = Time, y = value, col = name)) + geom_step() +
    theme_classic() + ylab("h(t)") + xlab("Time (years)") + ylim(0,1) +
    theme(legend.position = "none") + 
    scale_colour_manual(values = cbPalette[c(7,6,7)]) 
dat <- $d2 %>%
    data.frame() 
    colnames(dat) <- c("Time","Mean","LCI","UCI") 
p2 <- dat %>%
    pivot_longer(Mean:UCI) %>%
    ggplot(aes(x = Time, y = value, col = name)) + geom_step() +
    theme_classic() + ylab("h(t)") + xlab("Time (years)") + ylim(0,1) +
    theme(legend.position = "none") + 
    scale_colour_manual(values = cbPalette[c(7,6,7)])   
dat <- $d3 %>%
    data.frame() 
    colnames(dat) <- c("Time","Mean","LCI","UCI") 
p3 <- dat %>%
    pivot_longer(Mean:UCI) %>%
    ggplot(aes(x = Time, y = value, col = name)) + geom_step() +
    theme_classic() + ylab("h(t)") + xlab("Time (years)") + ylim(0,1) +
    theme(legend.position = "none") + 
    scale_colour_manual(values = cbPalette[c(7,6,7)]) 
dat <- $d4 %>%
    data.frame() 
    colnames(dat) <- c("Time","Mean","LCI","UCI") 
p4 <- dat %>%
    pivot_longer(Mean:UCI) %>%
    ggplot(aes(x = Time, y = value, col = name)) + geom_step() +
    theme_classic() + ylab("h(t)") + xlab("Time (years)") + ylim(0,1) +
    theme(legend.position = "none") + 
    scale_colour_manual(values = cbPalette[c(7,6,7)]) 
dat <- $d5 %>%
    data.frame() 
    colnames(dat) <- c("Time","Mean","LCI","UCI") 
p5 <- dat %>%
    pivot_longer(Mean:UCI) %>%
    ggplot(aes(x = Time, y = value, col = name)) + geom_step() +
    theme_classic() + ylab("h(t)") + xlab("Time (years)") + ylim(0,1) +
    theme(legend.position = "none") + 
    scale_colour_manual(values = cbPalette[c(7,6,7)]) 
dat <- $d6 %>%
    data.frame() 
    colnames(dat) <- c("Time","Mean","LCI","UCI") 
p6 <- dat %>%
    pivot_longer(Mean:UCI) %>%
    ggplot(aes(x = Time, y = value, col = name)) + geom_step() +
    theme_classic() + ylab("h(t)") + xlab("Time (years)") + ylim(0,1) +
    theme(legend.position = "none") + 
    scale_colour_manual(values = cbPalette[c(7,6,7)]) 
plot_grid(p1,p2,p3,p4,p5,p6, labels = c("1" ,"0.2", "0.01","1" ,"0.2", "0.01"))
"""

s1 = view(out4["Sk_x"], 1, 1:3, 1:500)
t1 = out4["Sk_t"][1:500]
R"""
df1 = data.frame($t1, t($s1))
colnames(df1) <- c("Time", "x1","x2", "x3")
df1 %>% 
    mutate(x2 = x1 + x2,
            x3 = x3 + x2) %>%
    pivot_longer(x2:x3) %>%
    ggplot(aes(x = Time, y = value, col = name)) + geom_line() + theme_classic() +
    theme(legend.position = "none") + xlab("") + ylab("Sampler time (arbitrary units)") +
    scale_colour_manual(values = cbPalette[c(6:7)])
ggsave($plotsdir("Ref.pdf"), width = 6, height = 4)
"""

Random.seed!(123)
df = CSV.read(datadir("colon.csv"), DataFrame)
y = df.years
maximum(y)
n = length(y)
breaks = collect(0.2:0.2:3.2)
p = 1
cens = df.status
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
t0 = 0.0
priors = BasicPrior(1.0, FixedV(1.0), FixedW(0.5), 1.0)
state0 = BPS(x0, v0, s0, t0, findall(s0))
nits = 1_000
nsmp = 100000
Random.seed!(123)
settings = Settings(nits, nsmp, 100000, 10, 0.0, 0.2, false, true)
@time out5 = pem_sample(state0, dat, priors, settings)


plot(out5["Smp_trans"][1,1,:])

@gif for i ∈ 1:2_800
    plot(vcat(0,breaks), vcat(mean(exp.(out5["Smp_trans"][1,:,i]), dims = 2), mean(exp.(out5["Smp_trans"][1,:,i]), dims = 2)[end]), 
        linetype=:steppost, ylim = (0,1))
end

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
@time out11 = pem_sample(state0, dat, priors, settings)
priors = BasicPrior(1.0, PC(0.2, 2, 0.1, 1, Inf), Beta(0.4, 10.0, 10.0), 1.0)
@time out2 = pem_sample(state0, dat, priors, settings)
@time out21 = pem_sample(state0, dat, priors, settings)
priors = BasicPrior(1.0, PC(0.2, 2, 1.0, 1, Inf), Beta(0.4, 10.0, 10.0), 1.0)
@time out3 = pem_sample(state0, dat, priors, settings)
@time out31 = pem_sample(state0, dat, priors, settings)

smps1 = out1["Smp_trans"]
s1 = view(smps1, 1, :, :)
plot(vcat(0,breaks), vcat(mean(exp.(s1), dims = 2), mean(exp.(s1), dims = 2)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.025),quantile.(eachrow(exp.(s1)), 0.025)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.25),quantile.(eachrow(exp.(s1)), 0.25)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.75),quantile.(eachrow(exp.(s1)), 0.75)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.975),quantile.(eachrow(exp.(s1)), 0.975)[end]),linetype=:steppost, ylim = (0,.6))

s1 = view(exp.(out1["Smp_trans"]), 1, :, :)
df = DataFrame(hcat(breaks, mean(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)
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
    theme(legend.position = "none") + scale_colour_manual(values = cbPalette[c(6,7,6,6,6)]) +
    scale_linetype_manual(values = c("dotdash","solid","dashed","dashed","dotdash")) + ylab("h(t)") + xlab("Time (years)")
    ggsave($plotsdir("Example.pdf"), width = 6, height = 4)
"""

smps1 = out11["Smp_trans"]
s1 = view(smps1, 1, :, :)
plot!(vcat(0,breaks), vcat(mean(exp.(s1), dims = 2), mean(exp.(s1), dims = 2)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.025),quantile.(eachrow(exp.(s1)), 0.025)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.25),quantile.(eachrow(exp.(s1)), 0.25)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.75),quantile.(eachrow(exp.(s1)), 0.75)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.975),quantile.(eachrow(exp.(s1)), 0.975)[end]),linetype=:steppost, ylim = (0,.6))

smps1 = out2["Smp_trans"]
s1 = view(smps1, 1, :, :)
plot(vcat(0,breaks), vcat(mean(exp.(s1), dims = 2), mean(exp.(s1), dims = 2)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.025),quantile.(eachrow(exp.(s1)), 0.025)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.25),quantile.(eachrow(exp.(s1)), 0.25)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.75),quantile.(eachrow(exp.(s1)), 0.75)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.975),quantile.(eachrow(exp.(s1)), 0.975)[end]),linetype=:steppost, ylim = (0,.6))

smps1 = out21["Smp_trans"]
s1 = view(smps1, 1, :, :)
plot!(vcat(0,breaks), vcat(mean(exp.(s1), dims = 2), mean(exp.(s1), dims = 2)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.025),quantile.(eachrow(exp.(s1)), 0.025)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.25),quantile.(eachrow(exp.(s1)), 0.25)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.75),quantile.(eachrow(exp.(s1)), 0.75)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.975),quantile.(eachrow(exp.(s1)), 0.975)[end]),linetype=:steppost, ylim = (0,.6))

smps1 = out3["Smp_trans"]
s1 = view(smps1, 1, :, :)
plot(vcat(0,breaks), vcat(mean(exp.(s1), dims = 2), mean(exp.(s1), dims = 2)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.025),quantile.(eachrow(exp.(s1)), 0.025)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.25),quantile.(eachrow(exp.(s1)), 0.25)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.75),quantile.(eachrow(exp.(s1)), 0.75)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.975),quantile.(eachrow(exp.(s1)), 0.975)[end]),linetype=:steppost, ylim = (0,.6))

smps1 = out31["Smp_trans"]
s1 = view(smps1, 1, :, :)
plot!(vcat(0,breaks), vcat(mean(exp.(s1), dims = 2), mean(exp.(s1), dims = 2)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.025),quantile.(eachrow(exp.(s1)), 0.025)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.25),quantile.(eachrow(exp.(s1)), 0.25)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.75),quantile.(eachrow(exp.(s1)), 0.75)[end]),linetype=:steppost)
plot!(vcat(0,breaks),vcat(quantile.(eachrow(exp.(s1)), 0.975),quantile.(eachrow(exp.(s1)), 0.975)[end]),linetype=:steppost, ylim = (0,.6))
