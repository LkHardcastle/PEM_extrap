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
    #ggsave($plotsdir("Extrap.png"), width = 9, height = 4)
    #ggsave($plotsdir("Extrap.pdf"), width = 14, height = 6)
"""

R"""
p <- $df %>%
    subset(x1 < 2.71) %>%
    ggplot() + #geom_smooth(se = F, col = cbPalette[7], linetype = "solid") + 
    theme_classic() + theme(text = element_text(size = 20)) + 
    xlab("Time (years)") + ylab("log h(t)") + ylim(0,0.7) + geom_vline(xintercept = 2.7, linetype = "dashed") + 
    annotate("segment", x = 0, y = 0.4, xend = 0.5, yend = 0.4, col = cbPalette[6], linetype = "solid") + 
    annotate("segment", x = 0.5, y = 0.5, xend = 1.2, yend = 0.5, col = cbPalette[6], linetype = "solid") + 
    annotate("segment", x = 1.2, y = 0.45, xend = 1.5, yend = 0.45, col = cbPalette[6], linetype = "solid") + 
    annotate("segment", x = 1.5, y = 0.3, xend = 1.7, yend = 0.3, col = cbPalette[6], linetype = "solid") + 
    annotate("segment", x = 1.7, y = 0.2, xend = 2, yend = 0.2, col = cbPalette[6], linetype = "solid") + 
    annotate("segment", x = 2, y = 0.1, xend = 2.2, yend = 0.1, col = cbPalette[6], linetype = "solid") + 
    annotate("segment", x = 2.2, y = 0.15, xend = 2.7, yend = 0.15, col = cbPalette[6], linetype = "solid") + 
    annotate("segment", x = 2.7, y = 0.15, xend = 3.5, yend = 0.15, col = cbPalette[7], linetype = "solid") + 
    annotate("segment", x = 3, y = 0.15, xend = 3.5, yend = 0.15, col = cbPalette[7], linetype = "solid") + 
    annotate("segment", x = 3.5, y = 0.15, xend = 4, yend = 0.15, col = cbPalette[7], linetype = "solid") +
    annotate("segment", x = 2.7, y = 0.22, xend = 3, yend = 0.22, col = cbPalette[7], linetype = "dashed") +
    annotate("segment", x = 3, y = 0.27, xend = 3.5, yend = 0.27, col = cbPalette[7], linetype = "dashed") + 
    annotate("segment", x = 3.5, y = 0.33, xend = 4, yend = 0.33, col = cbPalette[7], linetype = "dashed") +
    annotate("segment", x = 2.7, y = 0.08, xend = 3, yend = 0.08, col = cbPalette[7], linetype = "dashed") +
    annotate("segment", x = 3, y = 0.05, xend = 3.5, yend = 0.05, col = cbPalette[7], linetype = "dashed") + 
    annotate("segment", x = 3.5, y = 0.02, xend = 4, yend = 0.02, col = cbPalette[7], linetype = "dashed") + xlim(0,3.5)
p1 <- p + geom_vline(xintercept = c(0,0.5,1.2,1.5,1.7,2.2, 3), linetype = "dotted", alpha = 0.4)
ggsave($plotsdir("Talks","PWEXP1.png"), width = 10, height = 6)
p2 <- p + geom_vline(xintercept = seq(0,2.7,by = 0.1), linetype = "dotted", alpha = 0.4)
ggsave($plotsdir("Talks","PWEXP2.png"), width = 10, height = 6)
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
state0 = ECMC2(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0))
nits = 20
nsmp = 10000
settings = Settings(nits, nsmp, 1_000_000, 0.01,0.0, 0.1, false, true)
priors = BasicPrior(1.0, FixedV([1.0]), FixedW([0.5]), 1.0, Fixed(0.5),[RandomWalk()])
@time out3 = pem_sample(state0, dat, priors, settings)

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
ggsave($plotsdir("Talks","SM.png"), width = 12, height = 5)
"""

Random.seed!(12515)
n = 0
y = rand(Exponential(1.0),n)
breaks = collect(0.01:0.2:1.0)
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
state0 = ECMC2(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0))
nits = 100
nsmp = 10000
settings = Settings(nits, nsmp, 1_000_000, 0.01, 1.0, 0.1, false, true)
priors = BasicPrior(1.0, FixedV([1.0]), FixedW([0.5]), 1.0, Cts(5.0, 10.0, 1.0), [RandomWalk()])
@time out1 = pem_sample(state0, dat, priors, settings)

Skloc = out1["Sk_s_loc"]
loc_uniq = sort(unique(out1["Sk_s_loc"]))[1:(end-1)]
Skt = out1["Sk_t"]
Sks = out1["Sk_s"][1,:,:]
SkJ = vec(10 .- sum(isinf.(out1["Sk_s_loc"]), dims = 1))
class_ = fill(Inf, size(loc_uniq, 1), size(Skloc,2))


for i in axes(loc_uniq, 1)
    for j in axes(Skloc, 2)
        if loc_uniq[i] ∈ Skloc[:,j]
            if Sks[findfirst(Skloc[:,j] .== loc_uniq[i]), j] == 1
                    class_[i,j] = 1
            else
                class_[i,j] = 2
            end
        else
            class_[i,j] = 3
        end
    end
end

class_ = hcat(loc_uniq, class_)
df = DataFrame(class_, vcat("Location", string.(Skt)))

R"""
$df %>%
    pivot_longer(2:100, names_to = "Time", values_to = "Status") %>%
    mutate(Time = as.numeric(Time),
            Status = as.factor(Status),
            Location = as.numeric(Location)) %>%
    subset(Status != "3") %>%
    subset(Location > 0.01) %>%
    ggplot(aes(x = Time, y = Location, col = Status, group = Location)) + geom_line(size = 0.8) + theme_classic() +
    scale_colour_manual(labels = c("Unthinned", "Thinned"), values = cbPalette[6:7]) + theme(legend.title = element_blank(), legend.position = "bottom", text = element_text(size = 20)) + xlim(5,40) 
    ggsave($plotsdir("Talks","StickyInf.png"), width = 14, height = 4)
"""


x = -2:0.005:7
y1 = (1 .+ tanh.(x.*3.0)).*pdf.(Normal(0,1), x)
y2 = pdf.(Normal(3,1), x)

df = DataFrame(x = x, Barker = y1, EM = y2)

R"""
$df %>%
    pivot_longer(Barker:EM, names_to = "Method", values_to = "y") %>%
    ggplot(aes(x = x, y = y, col = Method)) + geom_line(size = 0.8) + theme_classic() +
    scale_colour_manual(values = cbPalette[6:7]) + xlab("theta") + ylab("density") + theme(legend.title = element_blank(), legend.position = "bottom", text = element_text(size = 20)) + 
    geom_vline(xintercept = 0, linetype = "dashed")
    ggsave($plotsdir("Talks","Discretisation.png"), width = 14, height = 4)
"""


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
    ggsave($plotsdir("Talks","Extrap.png"), width = 12, height = 5)
"""



df1 = CSV.read(datadir("ColonSmps","RW.csv"), DataFrame)
df2 = CSV.read(datadir("ColonSmps","Gaussian.csv"), DataFrame)
df3 = CSV.read(datadir("ColonSmps","Gamma.csv"), DataFrame)
df4 = CSV.read(datadir("ColonSmps","Gompertz.csv"), DataFrame)

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
dat_diffusion <- rbind(dat1, dat2, dat3, dat4)
"""

R"""
p1 <- dat_diffusion %>%
    subset(Time < 3.1) %>%
    pivot_longer(Mean:UCI,) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 10)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,0.5) + xlim(0,3)
p3 <- dat_diffusion %>%
    pivot_longer(c(Mean, Q1, Q4)) %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 10)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("solid","dotdash","dotdash")) + ylab("h(t)") + xlab("Time (years)") + ylim(0,1) + xlim(0,15)

plot_grid(p1,p3, nrow = 1)
"""