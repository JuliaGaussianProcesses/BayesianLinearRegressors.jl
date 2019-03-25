using BLR, Distributions, LinearAlgebra, Random, Test, Zygote, FDM
using BLR: BayesianLinearRegressor, posterior, marginals, cov, mean
using FDM: j′vp

@testset "BLR" begin
    include("bayesian_linear_regression.jl")
end
