using BayesianLinearRegressors, Distributions, LinearAlgebra, Random, Test, Zygote, FiniteDifferences
using BayesianLinearRegressors: BayesianLinearRegressor, posterior, marginals, cov, mean
using FiniteDifferences: j′vp

@testset "BayesianLinearRegressors" begin
    include("bayesian_linear_regression.jl")
end
