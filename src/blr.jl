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

"""
    IndexedBLR

Represents the random variables associated with `blr` at locations `X`. Really only
for internal use: you probably don't want to manually construct one of these in your code.
"""
struct IndexedBLR{Tblr<:BayesianLinearRegressor, TX<:AM}
    blr::Tblr
    X::TX
end
const IR = IndexedBLR

mean(ir::IR) = ir.X' * ir.blr.mw

function marginals(ir::IR)
    blr, X = ir.blr, ir.X
    α = cholesky(blr.Λw).U' \ X
    return Normal.(mean(ir), sqrt.(vec(sum(abs2, α; dims=1)) .+ blr.σ²))
end

function cov(ir::IR)
    blr, X = ir.blr, ir.X
    α = cholesky(blr.Λw).U' \ X
    return Symmetric(α' * α + blr.σ² * I)
end

# Internal utility functions.
function check_and_unpack(ir, y)
    @assert length(y) == size(ir.X, 2)
    return ir.blr, ir.X, length(y)
end
get_Σy(blr::BayesianLinearRegressor, N::Int) = Diagonal(fill(blr.σ², N))

(blr::BayesianLinearRegressor)(X::AM) = IndexedBLR(blr, X)

"""
    rand(ir::IR)
    rand(ir::IR, samples::Int)

Sample from the `BayesianLinearRegressor` `ir.blr` at `ir.X`. If `samples` is
provided then a `Matrix` of size `size(ir.X, 2) × samples` is produced where each column is
an independent sample. If `samples` is not provided then a `Vector` containing a single
sample is returned.
"""
function rand(rng::AbstractRNG, ir::IR, samples::Int)
    blr, X, D, N = ir.blr, ir.X, size(ir.X, 1), size(ir.X, 2)
    w = blr.mw .+ cholesky(blr.Λw).U \ randn(rng, D, samples)
    y = X' * w .+ cholesky(get_Σy(blr, N)).U' * randn(rng, N, samples)
end
rand(rng::AbstractRNG, ir::IR) = vec(rand(rng, ir, 1))

"""
    logpdf(ir::IR, y::AV{<:Real})

Compute the logpdf of observations `y` made at locations `ir.X` under `ir.blr`. i.e.
read `logpdf(f(X), y)`, where `f` is a `BayesianLinearRegressor`.
"""
function logpdf(ir::IR, y::AV{<:Real})
    blr, X, N = check_and_unpack(ir, y)

    A = cholesky(blr.Λw).U' \ X
    δy = y - X' * blr.mw

    # Compute stuff involving the observation noise.
    Σy = cholesky(get_Σy(blr, N))
    invUy_δy = Σy.U' \ δy
    α = A * (Σy.U \ invUy_δy)
    δyt_invΣy_δy = sum(abs2, invUy_δy)

    # Compute the posterior prediction.
    T = A / Σy.U
    Λεy = cholesky(Symmetric(T * T' + I))
    αt_invΛεy_α = sum(abs2, Λεy.U' \ α)

    # Compute the logpdf.
    return -(N * log(2π) + logdet(Λεy) + logdet(Σy) + δyt_invΣy_δy - αt_invΛεy_α) / 2
end

"""
    posterior(blr::BayesianLinearRegressor, X::AM{<:Real}, y::AV{<:Real})

Returns the posterior `BayesianLinearRegressor` produced by conditioning on `blr(X) = y`.
"""
function posterior(blr::BayesianLinearRegressor, X::AM{<:Real}, y::AV{<:Real}; σ²=nothing)
    @assert size(X, 2) == length(y)
    N = length(y)

    Uw = cholesky(blr.Λw).U
    A = Uw' \ X

    # Compute precision of the posterior over ε.
    Uy = cholesky(get_Σy(blr, N)).U
    T = A / Uy
    Λεy = Symmetric(T * T' + I)

    # Compute the mean of the posterior over ε.
    δy = y - X' * blr.mw
    α = T * (Uy' \ δy)
    mεy = cholesky(Λεy) \ α

    # Construct posterior BayesianLinearRegressor.
    Λεy_Uw = Λεy * Uw
    σ² = σ² === nothing ? blr.σ² : σ²
    return BayesianLinearRegressor(blr.mw + Λεy_Uw \ α, Symmetric(Uw' * Λεy_Uw), σ²)
end
