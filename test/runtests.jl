using BayesianLinearRegressors, Distributions, LinearAlgebra, Random, Test, Zygote, FDM
using BayesianLinearRegressors: BayesianLinearRegressor, posterior, marginals, cov, mean
using FDM: j′vp

@testset "BayesianLinearRegressors" begin
    include("bayesian_linear_regression.jl")
end
