"""
    IndexedRegressor

Represents the random variables associated with `blr` at locations `X`. Really only
for internal use: you probably don't want to manually construct one of these in your code.
"""
struct IndexedBLR{Tblr<:BayesianLinearRegressor, TX<:AM}
    blr::Tblr
    X::TX
end
const IR = IndexedRegressor

"""
    BayesianLinearRegressor{Tmw, TΛw, TΣy}

A Bayesian Linear Regressor is a distribution over linear functions with i.i.d.
homoscedastic additive aleatoric obserations noise, whose distribution is assumed zero-mean
and Gaussian.

# Fields
- `mw`: mean vector of the weights
- `Λw`: precision of the weights
- `σ²`: variance of noise
"""
struct BayesianLinearRegressor{Tmw<:AV, TΛw<:AM, Tσ²<:Real}
    mw::Tmw
    Λw::TΛw
    σ²::Tσ²
end
(blr::BayesianLinearRegressor)(X::AM) = IndexedRegressor(blr, X)

# Internal utility functions.
function check_and_unpack(ir, y)
    @assert length(y) == size(ir.X, 2)
    return ir.regressor, ir.X, length(y)
end
get_Σy(blr::BayesianLinearRegressor) = Diagonal(Fill(blr.σ²))

"""
    logpdf(ir::IR, y::AV{<:Real})

Compute the logpdf of observations `y` made at locations `ir.X` under `ir.regressor`. i.e.
read `logpdf(f(X), y)`, where `f` is a `BayesianLinearRegressor`.
"""
function logpdf(ir::IR, y::AV{<:Real})
    blr, X, N = check_and_unpack(ir, y)

    A = cholesky(blr.Λw).U' \ X
    δy = y - X' * blr.mw

    # Compute stuff involving the observation noise.
    Σy = cholesky(get_Σy(blr))
    invUy_δy = Σy.U' \ δy
    α = A * (Σy.U \ invUy_δy)
    δyt_invΣy_δy = sum(abs2, invUy_δy)

    # Compute the posterior prediction.
    Λwy = cholesky(A * (Σy \ A') + I)
    αt_invΛwy_α = sum(abs2, Λwy.U' \ α)

    # Compute the logpdf.
    return (N * log(2π) + logdet(Λwy) - logdet(Σy) + δyt_invΣy_δy - αt_invΛwy_α)
end

"""
    rand(ir::IR)
    rand(ir::IR, samples::Int)

Sample from the `BayesianLinearRegressor` `ir.regressor` at `ir.X`. If `samples` is
provided then a `Matrix` of size `size(ir.X, 2) × samples` is produced where each column is
an independent sample. If `samples` is not provided then a `Vector` containing a single
sample is returned.
"""
function rand(rng::AbstractRNG, ir::IR, samples::Int)
    blr, X, N = check_and_unpack(ir, y)
    w = blr.mw + cholesky(blr.Λw).U \ randn(rng, size(X, 1), samples)
    y = X' * w + cholesky(get_Σy(blr)).U' * randn(rng, N, samples)
end
rand(rng::AbstractRNG, ir::IR) = vec(rand(rng, ir, 1))

"""
    posterior(f::BayesianLinearRegressor, X::AM{<:Real}, y::AV{<:Real})

Returns the posterior `BayesianLinearRegressor` produced by conditioning on `f(X) = y`.
"""
function posterior(f::BayesianLinearRegressor, X::AM{<:Real}, y::AV{<:Real})
    blr, X, N = check_and_unpack(ir, y)

    Uw = cholesky(blr.Λw).U
    A = Uw' \ X

    # Compute the mean and precision of the posterior over ε.
    Σy = cholesky(get_Σy(blr))
    Λεy = A * (Σy \ A') + I
    mεy = cholesky(Λεy) \ (A * (Σy \ (y - X' * blr.mw)))

    # Construct posterior BayesianLinearRegressor.
    Uwt_Λεy = Uw' * Λεy
    return BayesianLinearRegressor(Uwt_Λεy * mεy, Uwt_Λεy * Uw)
end
