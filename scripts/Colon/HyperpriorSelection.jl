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

# Model 1

Random.seed!(7647)
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
nits = 200_000
nsmp = 10000
settings = Settings(nits, nsmp, 1_000_000, 0.5,0.5, 0.5, false, true)

drift_ = RandomWalk()

priors1 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [1.0], [9.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [drift_])
priors2 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [2.0], [8.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [drift_])
priors3 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [3.0], [7.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [drift_])
priors4 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [4.0], [6.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [drift_])
priors5 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [5.0], [5.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [drift_])
priors6 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [6.0], [4.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [drift_])
priors7 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [7.0], [3.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [drift_])
#priors8 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [1.0], [1.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [drift_])

# Save all models for plotting
Random.seed!(9102)
@time out1 = pem_sample(state0, dat, priors1, settings)
@time out2 = pem_sample(state0, dat, priors2, settings)
@time out3 = pem_sample(state0, dat, priors3, settings)
@time out4 = pem_sample(state0, dat, priors4, settings)
@time out5 = pem_sample(state0, dat, priors5, settings)
@time out6 = pem_sample(state0, dat, priors6, settings)
@time out7 = pem_sample(state0, dat, priors7, settings)
#@time out8 = pem_sample(state0, dat, priors8, settings)

DIC = Vector{Float64}()
push!(DIC, get_DIC(out1, dat)[3])
push!(DIC, get_DIC(out2, dat)[3])
push!(DIC, get_DIC(out3, dat)[3])
push!(DIC, get_DIC(out4, dat)[3])
push!(DIC, get_DIC(out5, dat)[3])
push!(DIC, get_DIC(out6, dat)[3])
push!(DIC, get_DIC(out7, dat)[3])
#push!(DIC, get_DIC(out8, dat)[3])
DIC
DIC1 = Vector{Float64}()
push!(DIC1, get_DIC(out1, dat)[4])
push!(DIC1, get_DIC(out2, dat)[4])
push!(DIC1, get_DIC(out3, dat)[4])
push!(DIC1, get_DIC(out4, dat)[4])
push!(DIC1, get_DIC(out5, dat)[4])
push!(DIC1, get_DIC(out6, dat)[4])
push!(DIC1, get_DIC(out7, dat)[4])
#push!(DIC1, get_DIC(out8, dat)[4])
DIC1
DIC .- DIC1
DIC2 = Vector{Float64}()
push!(DIC2, get_DIC(out1, dat)[2])
push!(DIC2, get_DIC(out2, dat)[2])
push!(DIC2, get_DIC(out3, dat)[2])
push!(DIC2, get_DIC(out4, dat)[2])
push!(DIC2, get_DIC(out5, dat)[2])
push!(DIC2, get_DIC(out6, dat)[2])
push!(DIC2, get_DIC(out7, dat)[2])
#push!(DIC2, get_DIC(out8, dat)[2])
DIC2

#CSV.write(datadir("ColonSmps","RWBeta.csv"), DataFrame(DIC = DIC))


# Model 2

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
nits = 500_000
nsmp = 20000
settings = Settings(nits, nsmp, 1_000_000, 0.5,0.5, 0.5, false, true)

drift_ = GaussLangevin(-1.0,1.0)

priors1 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [1.0], [9.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [drift_])
priors2 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [2.0], [8.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [drift_])
priors3 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [3.0], [7.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [drift_])
priors4 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [4.0], [6.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [drift_])
priors5 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [5.0], [5.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [drift_])
priors6 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [6.0], [4.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [drift_])
priors7 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [7.0], [3.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [drift_])
priors8 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [1.0], [1.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [drift_])

Random.seed!(9102)
DIC = Vector{Float64}()
@time out = pem_sample(state0, dat, priors1, settings)
push!(DIC, get_DIC(out, dat)[2])
@time out = pem_sample(state0, dat, priors2, settings)
push!(DIC, get_DIC(out, dat)[2])
@time out = pem_sample(state0, dat, priors3, settings)
push!(DIC, get_DIC(out, dat)[2])
@time out = pem_sample(state0, dat, priors4, settings)
push!(DIC, get_DIC(out, dat)[2])
@time out = pem_sample(state0, dat, priors5, settings)
push!(DIC, get_DIC(out, dat)[2])
@time out = pem_sample(state0, dat, priors6, settings)
push!(DIC, get_DIC(out, dat)[2])
@time out = pem_sample(state0, dat, priors7, settings)
push!(DIC, get_DIC(out, dat)[2])
@time out = pem_sample(state0, dat, priors8, settings)
push!(DIC, get_DIC(out, dat)[2])
CSV.write(datadir("ColonSmps","GaussBeta.csv"), DataFrame(DIC = DIC))

# Model 3

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
nits = 500_000
nsmp = 20000
settings = Settings(nits, nsmp, 1_000_000, 0.5,0.5, 0.5, false, true)

drift_ = GammaLangevin(0.5,2)

priors1 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [1.0], [9.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [drift_])
priors2 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [2.0], [8.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [drift_])
priors3 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [3.0], [7.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [drift_])
priors4 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [4.0], [6.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [drift_])
priors5 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [5.0], [5.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [drift_])
priors6 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [6.0], [4.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [drift_])
priors7 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [7.0], [3.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [drift_])
priors8 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [1.0], [1.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [drift_])


Random.seed!(9102)
DIC = Vector{Float64}()
@time out = pem_sample(state0, dat, priors1, settings)
push!(DIC, get_DIC(out, dat)[2])
@time out = pem_sample(state0, dat, priors2, settings)
push!(DIC, get_DIC(out, dat)[2])
@time out = pem_sample(state0, dat, priors3, settings)
push!(DIC, get_DIC(out, dat)[2])
@time out = pem_sample(state0, dat, priors4, settings)
push!(DIC, get_DIC(out, dat)[2])
@time out = pem_sample(state0, dat, priors5, settings)
push!(DIC, get_DIC(out, dat)[2])
@time out = pem_sample(state0, dat, priors6, settings)
push!(DIC, get_DIC(out, dat)[2])
@time out = pem_sample(state0, dat, priors7, settings)
push!(DIC, get_DIC(out, dat)[2])
@time out = pem_sample(state0, dat, priors8, settings)
push!(DIC, get_DIC(out, dat)[2])
CSV.write(datadir("ColonSmps","GammaBeta.csv"), DataFrame(DIC = DIC))
# Model 4

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
nits = 500_000
nsmp = 20000
settings = Settings(nits, nsmp, 1_000_000, 0.5,0.5, 0.5, false, true)

drift_ = GompertzBaseline(0.5)

priors1 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [1.0], [9.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [drift_])
priors2 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [2.0], [8.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [drift_])
priors3 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [3.0], [7.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [drift_])
priors4 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [4.0], [6.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [drift_])
priors5 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [5.0], [5.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [drift_])
priors6 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [6.0], [4.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [drift_])
priors7 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [7.0], [3.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [drift_])
priors8 = BasicPrior(1.0, PC([0.2], [2], [0.5], Inf), Beta([0.4], [1.0], [1.0]), 1.0, CtsPois(10.0, 50.0, 3.2), [drift_])

Random.seed!(9102)
DIC = Vector{Float64}()
@time out = pem_sample(state0, dat, priors1, settings)
push!(DIC, get_DIC(out, dat)[2])
@time out = pem_sample(state0, dat, priors2, settings)
push!(DIC, get_DIC(out, dat)[2])
@time out = pem_sample(state0, dat, priors3, settings)
push!(DIC, get_DIC(out, dat)[2])
@time out = pem_sample(state0, dat, priors4, settings)
push!(DIC, get_DIC(out, dat)[2])
@time out = pem_sample(state0, dat, priors5, settings)
push!(DIC, get_DIC(out, dat)[2])
@time out = pem_sample(state0, dat, priors6, settings)
push!(DIC, get_DIC(out, dat)[2])
@time out = pem_sample(state0, dat, priors7, settings)
push!(DIC, get_DIC(out, dat)[2])
@time out = pem_sample(state0, dat, priors8, settings)
push!(DIC, get_DIC(out, dat)[2])
CSV.write(datadir("ColonSmps","GompertzBeta.csv"), DataFrame(DIC = DIC))