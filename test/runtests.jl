using
    AbstractGPs,
    BayesianLinearRegressors,
    LinearAlgebra,
    PDMats,
    Random,
    Test

using BayesianLinearRegressors: BayesianLinearRegressor, posterior, marginals, cov, mean

include("test_utils.jl")

@testset "BayesianLinearRegressors" begin
    include("bayesian_linear_regression.jl")
    include("basis_function_regression.jl")
    include("sampling_functions.jl")
end
