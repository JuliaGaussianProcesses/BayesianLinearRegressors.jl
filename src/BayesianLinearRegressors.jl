module BayesianLinearRegressors

using AbstractGPs
using LinearAlgebra
using Random
using Statistics

using AbstractGPs: AbstractGP, _cholesky, FiniteGP

include("bayesian_linear_regression.jl")
include("basis_function_regression.jl")
include("sampling_functions.jl")

export logpdf, rand, mean, std, cov, BayesianLinearRegressor, marginals, posterior
export BasisFunctionRegressor

end # module
