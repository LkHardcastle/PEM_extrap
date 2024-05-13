using DrWatson
@quickactivate "PEM_extrap"
# For src
using DataStructures, LinearAlgebra, Distributions, Random
using Plots

include(srcdir("Sampler.jl"))
include(srcdir("PreProcessing.jl"))
include(srcdir("PostProcessing.jl"))

Random.seed!(123)
n = 0
y = rand(Exponential(1.0),n)
breaks = collect(0.25:0.25:1)
p = 1
cens = fill(1.0,n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
v_abs = vcat(1.0,[0.1,0.2,0.3])
x0, v0, s0 = init_params(p, dat, v_abs)
t0 = 0.0
priors = BasicPrior( 1.0, 0.0, 1.0)
nits = 1_000
nsmp = 1_000
settings = Settings(nits, nsmp, 10.0, 0.9, 0.5, 0.0, 0.0, v0, true)
dyn0 = ZigZag(1, 1, settings.tb_init, 0.0, 0.0, 0.0, 0.0, false, 0, 0.0, "Start", settings.v_abs,
                    SamplerEval(0,0,0,0,0,0,0))
Random.seed!(23653)
out1 = @time pem_sample(x0, s0, v0, t0, dat, priors, settings, dyn0)