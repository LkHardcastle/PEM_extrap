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
priors = BasicPrior(1.0, FixedV(1.0), FixedW(0.5), 0.0)
println(state0)
@time out3 = pem_sample(state0, dat, priors, settings)

x = [0.12657628020322645 -0.11775814209891318 0.1457155415673546 0.04090595202807143 0.04635033818928668 -0.06903190918753911 0.017443218941190844 -0.04591371159060143 0.02729823616640393 -0.028164206160436134]
a = [0.12657628020322645 0.00881813810431327 0.15453367967166787 0.1954396316997393 0.24178996988902599 0.17275806070148686 0.1902012796426777 0.14428756805207626 0.1715858042184802 0.14342159805804405]
w = [0.5 0.5 0.5 0.5 0.19927521949915716 0.0 0.0 0.0 0.0 0.0]
d = [0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0]

Ul = exp.(a).*w .- d.*a .+ (x.^2)./priors.σ.σ^2
sum(Ul)