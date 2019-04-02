"""
    BayesianLinearRegressor{Tmw, TΛw}

A Bayesian Linear Regressor is a distribution over linear functions given by
```julia
w ~ Normal(mw, Λw)
f(x) = dot(x, w)
```
where `mw` and `Λ` are the mean and precision of `w` respectively.
"""
struct BayesianLinearRegressor{Tmw<:AV, TΛw<:AM}
    mw::Tmw
    Λw::TΛw
end

"""
    IndexedBLR

Represents the random variables associated with `blr` at locations `X`. Really only
for internal use: you probably don't want to manually construct one of these in your code.
"""
struct IndexedBLR{Tblr<:BayesianLinearRegressor, TX<:AM, TΣy}
    blr::Tblr
    X::TX
    Σy::TΣy
end
const IR = IndexedBLR

(blr::BayesianLinearRegressor)(X::AM, Σy) = IndexedBLR(blr, X, Σy)
(blr::BayesianLinearRegressor)(X::AM, σ²::Real) = blr(X, Diagonal(fill(σ², size(X, 2))))
(blr::BayesianLinearRegressor)(x::AV, Σy) = blr(reshape(x, :, 1), Σy)

function marginals(ir::IR)
    blr, X = ir.blr, ir.X
    α = cholesky(blr.Λw).U' \ X
    return Normal.(mean(ir), sqrt.(vec(sum(abs2, α; dims=1)) .+ diag(ir.Σy)))
end
mean(ir::IR) = ir.X' * ir.blr.mw
function cov(ir::IR)
    blr, X = ir.blr, ir.X
    α = cholesky(blr.Λw).U' \ X
    return Symmetric(α' * α + ir.Σy)
end

# Internal utility function
function check_and_unpack(ir, y)
    @assert length(y) == size(ir.X, 2)
    return ir.blr, ir.X, length(y)
end

"""
    rand(rng::AbstractRNG, ir::IR)
    rand(rng::AbstractRNG, ir::IR, samples::Int)

Sample from the `BayesianLinearRegressor` `ir.blr` at `ir.X`. If `samples` is
provided then a `Matrix` of size `size(ir.X, 2) × samples` is produced where each column is
an independent sample. If `samples` is not provided then a `Vector` containing a single
sample is returned.
"""
function rand(rng::AbstractRNG, ir::IR, samples::Int)
    blr, X, D, N = ir.blr, ir.X, size(ir.X, 1), size(ir.X, 2)
    w = blr.mw .+ cholesky(blr.Λw).U \ randn(rng, D, samples)
    y = X' * w .+ cholesky(ir.Σy).U' * randn(rng, N, samples)
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
    Σy = cholesky(ir.Σy)
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
    posterior(ir::IR, y::AV{<:Real})

Returns the posterior `BayesianLinearRegressor` produced by conditioning on
`ir.blr(ir.X) = y`, from which all posterior predictive qtts can be obtained.
"""
function posterior(ir::IR, y::AV{<:Real})
    @assert size(ir.X, 2) == length(y)
    N, blr = length(y), ir.blr

    Uw = cholesky(blr.Λw).U
    A = Uw' \ ir.X

    # Compute precision of the posterior over ε.
    Uy = cholesky(ir.Σy).U
    T = A / Uy
    Λεy = Symmetric(T * T' + I)

    # Compute the mean of the posterior over ε.
    δy = y - mean(ir)
    α = T * (Uy' \ δy)
    mεy = cholesky(Λεy) \ α

    # Construct posterior BayesianLinearRegressor.
    Λεy_Uw = Λεy * Uw
    return BayesianLinearRegressor(blr.mw + Λεy_Uw \ α, Symmetric(Uw' * Λεy_Uw))
end
posterior(ir::IR, y::Real) = posterior(ir, [y])

# Temporary implementation of `fill` while Zygote can't handle it.
Zygote.@adjoint fill(x::Real, dims...) = fill(x, dims...), Δ->(sum(Δ), map(_->nothing, dims)...)
