using BLR, Distributions, LinearAlgebra, Random, Test
using BLR: BayesianLinearRegressor

@testset "BLR" begin
    include("bayesian_linear_regression.jl")
end
