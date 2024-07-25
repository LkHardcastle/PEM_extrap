using DrWatson
@quickactivate "PEM_extrap"
# For src
using DataStructures, LinearAlgebra, Distributions, Random, Optim, Roots, SpecialFunctions

include(srcdir("Sampler.jl"))
include(srcdir("PreProcessing.jl"))
include(srcdir("PostProcessing.jl"))

Random.seed!(12515)
n = 1
y = rand(Exponential(1.0),n)
breaks = collect(0.5:0.5:5)
p = 1
cens = fill(1.0,n)
covar = fill(1.0, 1, n)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
t0 = 0.0

Random.seed!(34734)
dat = init_data(y, cens, covar, breaks)
x0, v0, s0 = init_params(p, dat)
v0 = v0./norm(v0)
state0 = ECMC2(x0, v0, s0, t0, true, findall(s0))
nits = 10_000
nsmp = 100000
settings = Settings(nits, nsmp, 1_000_000, 0.5,0.0, 0.1, false, true)
println(state0)
priors = BasicPrior(1.0, FixedV(1.0), FixedW(0.5), 0.0)
@time out3 = pem_sample(state0, dat, priors, settings)

state = ECMC2([0.12657628020322645 -0.11775814209891318 0.1457155415673546 0.04090595202807143 0.04635033818928668 -0.06903190918753911 0.017443218941190844 -0.04591371159060143 0.02729823616640393 -0.028164206160436134], [0.22113559276063754 -0.4039811263027777 0.030441310788867543 -0.3024634048823738 -0.29272292179597437 0.2692356804989296 -0.38925359243687824 -0.19073822203424093 0.5334764889310573 0.2545905754332952], Bool[1 1 1 1 1 1 1 1 1 1], 0.0, true, CartesianIndex{2}[CartesianIndex(1, 1), CartesianIndex(1, 2), CartesianIndex(1, 3), CartesianIndex(1, 4), CartesianIndex(1, 5), CartesianIndex(1, 6), CartesianIndex(1, 7), CartesianIndex(1, 8), CartesianIndex(1, 9), CartesianIndex(1, 10)])
(0.031829680024050656, 0.04483243652894534, 1.0032345597246526)