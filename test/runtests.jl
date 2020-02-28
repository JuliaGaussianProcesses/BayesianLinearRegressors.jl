using BayesianLinearRegressors, Distributions, LinearAlgebra, Random, Test, Zygote, FiniteDifferences
using BayesianLinearRegressors: BayesianLinearRegressor, posterior, marginals, cov, mean
using FiniteDifferences: j′vp

# Hack to make sure that we can ignore random number generation when computing derivatives.
Zygote.@nograd MersenneTwister

@testset "BayesianLinearRegressors" begin
    include("bayesian_linear_regression.jl")
end
