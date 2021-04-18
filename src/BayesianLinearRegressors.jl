module BayesianLinearRegressors

using AbstractGPs
using LinearAlgebra
using Random
using Statistics

using AbstractGPs: AbstractGP, _cholesky, FiniteGP

include("bayesian_linear_regression.jl")

export logpdf, rand, mean, std, cov, BayesianLinearRegressor, marginals, posterior

end # module
