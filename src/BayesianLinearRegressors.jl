module BayesianLinearRegressors

using Random, LinearAlgebra, Distributions, Statistics, Zygote

import Statistics: mean, cov
import Random: rand
import Distributions: logpdf

const AV = AbstractVector
const AM = AbstractMatrix

# Hack to make sure that we can ignore random number generation when computing derivatives.
Zygote.@nograd MersenneTwister

include("bayesian_linear_regression.jl")

end # module
