using AbstractGPs
using BayesianLinearRegressors
using Distributions
using FiniteDifferences
using LinearAlgebra
using Random
using Test
using Zygote
using PDMats

using BayesianLinearRegressors: BayesianLinearRegressor, posterior, marginals, cov, mean
using FiniteDifferences: j′vp

include("test_utils.jl")

@testset "BayesianLinearRegressors" begin
    include("bayesian_linear_regression.jl")
    include("basis_function_regression.jl")
    include("sampling_functions.jl")
end
