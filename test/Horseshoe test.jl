using DrWatson
@quickactivate "PEM_extrap"
# For src
using DataStructures, LinearAlgebra, Distributions, Random
using Plots, CSV, DataFrames

include(srcdir("Sampler.jl"))
include(srcdir("PreProcessing.jl"))
include(srcdir("PostProcessing.jl"))

df = CSV.read(datadir("colon.csv"), DataFrame)
y = df.time./365
maximum(y)
n = length(y)
trunc_ind = findall(y .> 3.0)
y[trunc_ind] .= 3.0
breaks = collect(0.15:0.15:3.15)
p = 1
cens = df.status
cens[trunc_ind] .= 0.0
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
v_abs = vcat(1.0,collect(0.05:0.05:1))
#v_abs = vcat(1.0,collect(0.1:0.1:0.6))
x0, v0, s0 = init_params(p, dat, v_abs)
t0 = 0.0
nits = 100_000
nsmp = 10_000
settings = Settings(nits, nsmp, 0.9, 0.5, 1.0, v0, false)

Random.seed!(23653)
priors = HyperPrior2(fill(0.4, size(x0)), 0.5, 20.0, 20.0, 0.1, 1.0, 0.0,1.0,1.0, 0.0)
out1 = @time pem_sample(x0, s0, v0, t0, dat, priors, settings)

Random.seed!(23653)
priors = HyperPrior3(fill(0.4, size(x0)), 0.5, 20.0, 20.0, 
                    ones(1,length(breaks) - 1), 1.0, 
                    1.0,0.0,
                    1.0, ones(1,length(breaks) - 1), ones(1,length(breaks) - 1),
                    0.0)
out2 = @time pem_sample(x0, s0, v0, t0, dat, priors, settings)