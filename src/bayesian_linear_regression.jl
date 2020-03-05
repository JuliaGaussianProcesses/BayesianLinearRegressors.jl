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

# Computation utilised in both `logpdf` and `posterior`.
function __compute_inference_quantities(ir::IR, y::AV{<:Real})
    blr, X, N = check_and_unpack(ir, y)

    Uw = cholesky(blr.Λw).U
    Σy = cholesky(ir.Σy)

    Bt = Σy.U' \ (Uw' \ X)'
    δy = Σy.U' \ (y - mean(ir))

    logpdf_δy = -(N * log(2π) + logdet(Σy) + sum(abs2, δy)) / 2

    Λεy = cholesky(Symmetric(Bt'Bt + I))

    return Uw, Bt, δy, logpdf_δy, Λεy
end

"""
    logpdf(ir::IR, y::AV{<:Real})

Compute the logpdf of observations `y` made at locations `ir.X` under `ir.blr`. i.e.
read `logpdf(f(X), y)`, where `f` is a `BayesianLinearRegressor`.
"""
function logpdf(ir::IR, y::AV{<:Real})
    _, Bt, δy, logpdf_δy, Λεy = __compute_inference_quantities(ir, y)
    return -(logdet(Λεy) - sum(abs2, Λεy.U' \ (Bt'δy))) / 2 + logpdf_δy
end

"""
    posterior(ir::IR, y::AV{<:Real})

Returns the posterior `BayesianLinearRegressor` produced by conditioning on
`ir.blr(ir.X) = y`, from which all posterior predictive qtts can be obtained.
"""
function posterior(ir::IR, y::AV{<:Real})
    Uw, Bt, δy, _, Λεy = __compute_inference_quantities(ir, y)

    # Compute posterior over decorrelated weights.
    mεy = Λεy \ (Bt'δy)

    # Compute posterior over weights.
    T = Λεy.U * Uw
    Λwy = Symmetric(T'T)
    mwy = ir.blr.mw + Uw \ mεy
    return BayesianLinearRegressor(mwy, Λwy)
end
posterior(ir::IR, y::Real) = posterior(ir, [y])
