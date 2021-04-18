using AbstractGPs
using BayesianLinearRegressors
using Distributions
using FiniteDifferences
using LinearAlgebra
using Random
using Test
using Zygote

using BayesianLinearRegressors: BayesianLinearRegressor, posterior, marginals, cov, mean
using FiniteDifferences: j′vp

@testset "BayesianLinearRegressors" begin
    include("bayesian_linear_regression.jl")
end
