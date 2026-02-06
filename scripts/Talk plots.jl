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
    ggsave($plotsdir("Extrap_square.png"), width = 6, height = 6)
    #ggsave($plotsdir("Extrap.pdf"), width = 14, height = 6)
"""

R"""
p <- $df %>%
    subset(x1 < 2.71) %>%
    ggplot() + #geom_smooth(se = F, col = cbPalette[7], linetype = "solid") + 
    theme_classic() + theme(text = element_text(size = 20)) + 
    xlab("Time (years)") + ylab("log h(y)") + ylim(0,0.7) + geom_vline(xintercept = 2.7, linetype = "dashed") + 
    annotate("segment", x = 0, y = 0.4, xend = 0.5, yend = 0.4, col = cbPalette[6], linetype = "solid", size = 0.8) + 
    annotate("segment", x = 0.5, y = 0.5, xend = 1.2, yend = 0.5, col = cbPalette[6], linetype = "solid", size = 0.8) + 
    annotate("segment", x = 1.2, y = 0.45, xend = 1.5, yend = 0.45, col = cbPalette[6], linetype = "solid", size = 0.8) + 
    annotate("segment", x = 1.5, y = 0.3, xend = 1.7, yend = 0.3, col = cbPalette[6], linetype = "solid", size = 0.8) + 
    annotate("segment", x = 1.7, y = 0.2, xend = 2, yend = 0.2, col = cbPalette[6], linetype = "solid", size = 0.8) + 
    annotate("segment", x = 2, y = 0.1, xend = 2.2, yend = 0.1, col = cbPalette[6], linetype = "solid", size = 0.8) + 
    annotate("segment", x = 2.2, y = 0.15, xend = 2.7, yend = 0.15, col = cbPalette[6], linetype = "solid", size = 0.8) + 
    annotate("segment", x = 2.7, y = 0.15, xend = 3.5, yend = 0.15, col = cbPalette[7], linetype = "solid", size = 0.8) + 
    annotate("segment", x = 3, y = 0.15, xend = 3.5, yend = 0.15, col = cbPalette[7], linetype = "solid", size = 0.8) + 
    annotate("segment", x = 3.5, y = 0.15, xend = 4, yend = 0.15, col = cbPalette[7], linetype = "solid", size = 0.8) +
    annotate("segment", x = 2.7, y = 0.22, xend = 3, yend = 0.22, col = cbPalette[7], linetype = "dashed", size = 0.8) +
    annotate("segment", x = 3, y = 0.27, xend = 3.5, yend = 0.27, col = cbPalette[7], linetype = "dashed", size = 0.8) + 
    annotate("segment", x = 3.5, y = 0.33, xend = 4, yend = 0.33, col = cbPalette[7], linetype = "dashed", size = 0.8) +
    annotate("segment", x = 2.7, y = 0.08, xend = 3, yend = 0.08, col = cbPalette[7], linetype = "dashed", size = 0.8) +
    annotate("segment", x = 3, y = 0.05, xend = 3.5, yend = 0.05, col = cbPalette[7], linetype = "dashed", size = 0.8) + 
    annotate("segment", x = 3.5, y = 0.02, xend = 4, yend = 0.02, col = cbPalette[7], linetype = "dashed", size = 0.8) + xlim(0,3.5)
p1 <- p + geom_vline(xintercept = c(0,0.5,1.2,1.5,1.7,2.2, 3), linetype = "dotted", alpha = 0.4)
ggsave($plotsdir("Talks","PWEXP1.png"), width = 10, height = 6)
p2 <- p + geom_vline(xintercept = seq(0,3.5,by = 0.1), linetype = "dotted", alpha = 0.8)
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
    ggsave($plotsdir("Talks","StickyInf.pdf"), width = 14, height = 5)
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
    ggsave($plotsdir("Talks","Discretisation.pdf"), width = 14, height = 5)
"""


t = collect(0.3:0.3:3.9)
ht = [0.3,0.9,0.6,0.5,0.3,0.1,0.1,0.1,0.15,0.16,0.2,0.25,0.3]

df = DataFrame(hcat(t, ht, vcat(fill(1,8),fill(2,5))), :auto)

R"""
$df %>%
    subset(x1 < 2.71) %>%
    ggplot(aes(x = x1, y = x2)) + geom_smooth(se = F, col = cbPalette[7]) + theme_classic() +
    theme(text = element_text(size = 20)) +
    xlab("Time (years)") + ylab("h(y)") + ylim(0,0.7) + geom_vline(xintercept = 2.7, linetype = "dashed") + 
    annotate("segment", x = 2.7, y = 0.15, xend = 4, yend = 0.4, col = cbPalette[6], linetype = "dashed") + 
    annotate("segment", x = 2.7, y = 0.15, xend = 4, yend = 0.12, col = cbPalette[4], linetype = "dashed") + 
    annotate("segment", x = 2.7, y = 0.15, xend = 4, yend = 0.7, col = cbPalette[7], linetype = "dashed")
    #ggsave($plotsdir("Talks","Extrap.png"), width = 12, height = 5)
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
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("dotdash","solid","dotdash","dotdash","dotdash")) + ylab("h(y)") + xlab("Time (years)") + ylim(0,0.5) + xlim(0,3) + 
    geom_vline(xintercept = 3, linetype = "dotted")
p2 <- dat_diffusion %>%
    pivot_longer(c(Mean, Q1, Q4)) %>%
    ggplot(aes(x = Time, y = log(value), col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "bottom", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8,4,6,7)]) +
    scale_linetype_manual(values = c("solid","dotdash","dotdash")) + ylab("h(y)") + xlab("Time (years)") + ylim(0,1) + xlim(0,15)  + 
    geom_vline(xintercept = 3, linetype = "dotted")
p3 <- dat_diffusion %>%
    pivot_longer(c(Mean, Q1, Q4)) %>%
    subset(Model == "Random Walk") %>%
    ggplot(aes(x = Time, y = log(value), col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(7)]) +
    scale_linetype_manual(values = c("solid","dotdash","dotdash")) + ylab("logh(y)") + xlab("Time (years)") + #ylim(0,1) + xlim(0,15)  + 
    geom_vline(xintercept = 3, linetype = "dotted")
p4 <- dat_diffusion %>%
    pivot_longer(c(Mean, Q1, Q4)) %>%
    subset(Model == "Log-Normal Langevin") %>%
    ggplot(aes(x = Time, y = log(value), col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6)]) +
    scale_linetype_manual(values = c("solid","dotdash","dotdash")) + ylab("logh(y)") + xlab("Time (years)") + #ylim(0,1) + xlim(0,15)  + 
    geom_vline(xintercept = 3, linetype = "dotted")
p5 <- dat_diffusion %>%
    pivot_longer(c(Mean, Q1, Q4)) %>%
    subset(Model == "Gamma Langevin") %>%
    ggplot(aes(x = Time, y = log(value), col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(8)]) +
    scale_linetype_manual(values = c("solid","dotdash","dotdash")) + ylab("logh(y)") + xlab("Time (years)") + #ylim(0,1) + xlim(0,15)  + 
    geom_vline(xintercept = 3, linetype = "dotted")
p6 <- dat_diffusion %>%
    pivot_longer(c(Mean, Q1, Q4)) %>%
    subset(Model == "Gompertz dynamics") %>%
    ggplot(aes(x = Time, y = log(value), col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 2), linetype = FALSE) + 
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(4)]) +
    scale_linetype_manual(values = c("solid","dotdash","dotdash")) + ylab("loghyt)") + xlab("Time (years)") + #ylim(0,1) + xlim(0,15)  + 
    geom_vline(xintercept = 3, linetype = "dotted")

p_extrap <- plot_grid(p3,p4,p5,p6, ncol = 2)
grobs <- ggplotGrob(p2)$grobs
legend <- grobs[[which(sapply(grobs, function(x) x$name) == "guide-box")]]
p <- plot_grid(p1,p_extrap, nrow = 1)
p_ <- plot_grid(p, legend, nrow = 2,  rel_heights = c(1, .1))

#ggsave($plotsdir("Talks","Colon.png"), width = 14, height = 6)
ggsave($plotsdir("Talks","Colon.pdf"), width = 14, height = 6)
"""


df1 = CSV.read(datadir("EM_exp1.csv"), DataFrame)
df2 = CSV.read(datadir("EM_exp2.csv"), DataFrame)
R"""
p1 <- $df1 %>%
    pivot_longer(c(EM1:Barker5)) %>%
    mutate(method = case_when(
        grepl("EM", name, fixed = TRUE) ~ "Euler-Maruyama",
        grepl("Barker", name, fixed = TRUE) ~ "Barker"
            ),
            step_size = case_when(
                grepl("1", name, fixed = TRUE) ~ "0.01",
                grepl("2", name, fixed = TRUE) ~ "0.05",
                grepl("3", name, fixed = TRUE) ~ "0.1",
                grepl("4", name, fixed = TRUE) ~ "0.25",
                grepl("5", name, fixed = TRUE) ~ "0.5"
            )) %>%
    ggplot(aes(x = step_size, y = value, col = method)) + geom_boxplot(size = 0.5) +
    theme_classic() + scale_colour_manual(values = cbPalette[6:7]) + geom_hline(yintercept = 0.5, linetype = "dotted") + ylim(0,1) +
    theme(legend.position = "none", text = element_text(size = 20)) + ylab("omega") + xlab("Step size")

p2 <- $df2 %>%
    pivot_longer(c(EM1:Barker5)) %>%
    mutate(method = case_when(
        grepl("EM", name, fixed = TRUE) ~ "Euler-Maruyama",
        grepl("Barker", name, fixed = TRUE) ~ "Barker"
            ),
            step_size = case_when(
                grepl("1", name, fixed = TRUE) ~ "0.01",
                grepl("2", name, fixed = TRUE) ~ "0.05",
                grepl("3", name, fixed = TRUE) ~ "0.1",
                grepl("4", name, fixed = TRUE) ~ "0.25",
                grepl("5", name, fixed = TRUE) ~ "0.5"
            )) %>%
    ggplot(aes(x = step_size, y = value, col = method)) + geom_boxplot(size = 0.5) +
    theme_classic() + scale_colour_manual(values = cbPalette[6:7]) + geom_hline(yintercept = 0.5, linetype = "dotted") + ylim(0,1) + 
    theme(legend.position = "none", text = element_text(size = 20)) + ylab("omega") + xlab("Step size")
p3 <- $df2 %>%
    pivot_longer(c(EM1:Barker5)) %>%
    mutate(Discretisation = case_when(
        grepl("EM", name, fixed = TRUE) ~ "Euler-Maruyama",
        grepl("Barker", name, fixed = TRUE) ~ "Barker"
            ),
            step_size = case_when(
                grepl("1", name, fixed = TRUE) ~ "0.01",
                grepl("2", name, fixed = TRUE) ~ "0.05",
                grepl("3", name, fixed = TRUE) ~ "0.1",
                grepl("4", name, fixed = TRUE) ~ "0.25",
                grepl("5", name, fixed = TRUE) ~ "0.5"
            )) %>%
    ggplot(aes(x = step_size, y = value, col = Discretisation)) + geom_boxplot(size = 0.5) +
    theme_classic() + scale_colour_manual(values = cbPalette[6:7]) + geom_hline(yintercept = 0.5, linetype = "dotted") + ylim(0,1) + 
    theme(legend.position = "bottom", text = element_text(size = 20))

p <- plot_grid(p2,p1, ncol = 2)
grobs <- ggplotGrob(p3)$grobs
legend <- grobs[[which(sapply(grobs, function(x) x$name) == "guide-box")]]
p_ <- plot_grid(p, legend, nrow = 2,  rel_heights = c(1, .1))
ggsave($plotsdir("Talks","Disc_exp.png"), width = 14, height = 4.8)
"""



Random.seed!(12515)
n = 0
y = rand(Exponential(1.0),n)
breaks = collect(1:1:300)
p = 1
cens = fill(1.0,n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0
state0 = ECMC2(x0, v0, s0, fill(false, size(s0)), breaks, t0, length(breaks),  true, findall(s0))
nits = 50_000
nsmp = 20_000

Random.seed!(23462)
settings = Settings(nits, nsmp, 1_000_000, 2.0, 2.0, 1.0, false, true)
priors1 = BasicPrior(1.0, FixedV([0.2]), FixedW([0.5]), 0.0, Fixed(0.1), [RandomWalk()])
@time out1 = pem_sample(state0, dat, priors1, settings)
priors2 = BasicPrior(1.0, FixedV([0.2]), FixedW([0.5]), 0.0, Fixed(0.1), [GaussLangevin(2.0,1.0)])
@time out2 = pem_sample(state0, dat, priors2, settings)
priors3 = BasicPrior(1.0, FixedV([0.2]), FixedW([0.5]), 0.0, Fixed(0.1), [GompertzBaseline(0.1)])
@time out3 = pem_sample(state0, dat, priors3, settings)

s1 = view(exp.(cumsum(out1["Smp_x"], dims = 2)), 1, :, :)
df1 = DataFrame(hcat(breaks, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

s1 = view(exp.(cumsum(out2["Smp_x"], dims = 2)), 1, :, :)
df2 = DataFrame(hcat(breaks, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

s1 = view(exp.(cumsum(out3["Smp_x"], dims = 2)), 1, :, :)
df3 = DataFrame(hcat(breaks, median(s1, dims = 2), quantile.(eachrow(s1), 0.025), quantile.(eachrow(s1), 0.25), quantile.(eachrow(s1), 0.75), quantile.(eachrow(s1), 0.975)), :auto)

R"""
dat1 = data.frame($df1)
dat1 = cbind(dat1, "Random Walk")
colnames(dat1) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat2 = data.frame($df2)
dat2 = cbind(dat2, "Log-Normal stationary")
colnames(dat2) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model") 
dat3 = data.frame($df3)
dat3 = cbind(dat3, "Gompertz dynamics")
colnames(dat3) <- c("Time","Mean","LCI","Q1","Q4","UCI","Model")  
dat_diffusion <- rbind(dat1, dat2, dat3)
"""

R"""
p1 <- dat_diffusion %>%
    pivot_longer(c(Mean, Q1, Q4)) %>%
    subset(Model == "Random Walk") %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 1), linetype = FALSE) + 
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(6)]) +
    scale_linetype_manual(values = c("solid","dotdash","dotdash")) + ylab("h(y)") + xlab("Time (arbitrary units)") + ylim(0,17)
p2 <- dat_diffusion %>%
    pivot_longer(c(Mean, Q1, Q4)) %>%
    subset(Model == "Log-Normal stationary") %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 1), linetype = FALSE) + 
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(7)]) +
    scale_linetype_manual(values = c("solid","dotdash","dotdash")) + ylab("h(y)") + xlab("Time (arbitrary units)") + ylim(0,17)
p3 <- dat_diffusion %>%
    pivot_longer(c(Mean, Q1, Q4)) %>%
    subset(Model == "Gompertz dynamics") %>%
    ggplot(aes(x = Time, y = value, col = Model, linetype = name)) + geom_step() +
    theme_classic() + guides(col = guide_legend(nrow = 1), linetype = FALSE) + 
    theme(legend.position = "none", text = element_text(size = 20)) + scale_colour_manual(values = cbPalette[c(4)]) +
    scale_linetype_manual(values = c("solid","dotdash","dotdash")) + ylab("h(y)") + xlab("Time (arbitrary units)") + ylim(0,17)
plot_grid(p1,p2,p3,ncol = 3)
#ggsave($plotsdir("Talks","Diffusions.png"), width = 17, height = 4.8)
ggsave($plotsdir("Talks","Diffusions.pdf"), width = 14, height = 4.5)
"""


### Simple parametric

R"""
y <- seq(0.01,5,0.01)
lambda <- 0.5
gamma <- 0.6
h_y <- lambda*gamma*y^(gamma-1)
df <- data.frame(Time = y, h_y = h_y)
df %>%
  ggplot(aes(x = Time, y = h_y)) + geom_path(col = cbPalette[6]) + 
  theme_classic() + ylab("h(y)") + ylim(0,1) + theme(text = element_text(size = 20)) + 
  geom_vline(xintercept = 3, linetype = "dotted")
ggsave($plotsdir("Talks","Parametric.png"), width = 12, height = 6)
"""

R"""
set.seed(8090)
t <- seq(0,3,0.001)
innovations <- rnorm(length(t),0,sqrt(0.001))
brownian_motion <- c(cumsum(innovations))
plot(t, brownian_motion, type = "lty")
discretisation <- rep(NA,length(brownian_motion))
for(i in seq(1,length(brownian_motion)- 300,300)){
    discretisation[i:(i+299)] <- brownian_motion[i]
}
plot_data <- data.frame(Time = t, Diffusion = brownian_motion, Discretisation = discretisation)
plot_data %>%
    pivot_longer(Diffusion:Discretisation, values_to = "Process") %>%
    subset(name == "Diffusion") %>%
    ggplot(aes(x = Time, y = Process, col = name)) + geom_line() + 
    theme_classic() + scale_colour_manual(values = cbPalette[6]) + 
  theme(legend.position="none") +
  theme(legend.title=element_blank(), text = element_text(size = 20))
  ggsave($plotsdir("Talks","Diffusion1.png"), width = 8  , height = 6)
plot_data %>%
    pivot_longer(Diffusion:Discretisation, values_to = "Process") %>%
    ggplot(aes(x = Time, y = Process, col = name)) + geom_line() + 
    theme_classic() + scale_colour_manual(values = cbPalette[6:7]) + 
  theme(legend.position="none") +
  theme(legend.title=element_blank(), text = element_text(size = 20))
  ggsave($plotsdir("Talks","Diffusion2.png"), width = 8, height = 6)
"""