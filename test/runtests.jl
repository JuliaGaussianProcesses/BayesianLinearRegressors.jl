using BLR, Distributions, LinearAlgebra, Random, Test
using BLR: BayesianLinearRegressor, posterior, marginals, cov, mean

@testset "BLR" begin
    include("bayesian_linear_regression.jl")
end
