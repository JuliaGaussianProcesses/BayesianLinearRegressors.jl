module BayesianLinearRegressors

using Random, LinearAlgebra, Distributions, Statistics

import Statistics: mean, cov
import Random: rand
import Distributions: logpdf

const AV = AbstractVector
const AM = AbstractMatrix

export logpdf, rand, mean, std, cov, BayesianLinearRegressor, marginals, posterior

include("bayesian_linear_regression.jl")

end # module
