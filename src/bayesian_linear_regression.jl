"""
    BayesianLinearRegressor{Tmw, TΛw}

A Bayesian Linear Regressor is a distribution over linear functions given by
```julia
w ~ Normal(mw, Λw)
f(x) = dot(x, w)
```
where `mw` and `Λw` are the mean and precision of `w`, respectively.
"""
struct BayesianLinearRegressor{Tmw<:AbstractVector,TΛw<:AbstractMatrix,Tϕ} <: AbstractGP
    mw::Tmw
    Λw::TΛw
    ϕ::Tϕ
end

function BayesianLinearRegressor(mw::AbstractVector, Λw::AbstractMatrix)
    return BayesianLinearRegressor(mw, Λw, identity)
end

function (blr::BayesianLinearRegressor)(x::AbstractVector, args...)
    return FiniteGP(blr, blr.ϕ(x), args...)
end

const FiniteBLR = FiniteGP{<:BayesianLinearRegressor}

# All code below implements the primary + secondary AbstractGPs.jl APIs.

x_as_colvecs(fx::FiniteBLR) = x_as_colvecs(fx.x)

x_as_colvecs(x::ColVecs) = x

x_as_colvecs(x::RowVecs) = ColVecs(x.X')

function x_as_colvecs(x::T) where {T<:AbstractVector}
    return error(
        "$T is not a subtype of AbstractVector that is known. Please provide either a",
        "ColVecs or RowVecs.",
    )
end

AbstractGPs.mean(fx::FiniteBLR) = x_as_colvecs(fx).X' * fx.f.mw

function AbstractGPs.cov(fx::FiniteBLR)
    α = _cholesky(fx.f.Λw).U' \ x_as_colvecs(fx).X
    return Symmetric(α' * α + fx.Σy)
end

function AbstractGPs.var(fx::FiniteBLR)
    α = _cholesky(fx.f.Λw).U' \ x_as_colvecs(fx).X
    return vec(sum(abs2, α; dims=1)) .+ diag(fx.Σy)
end

AbstractGPs.mean_and_cov(fx::FiniteBLR) = (mean(fx), cov(fx))

AbstractGPs.mean_and_var(fx::FiniteBLR) = (mean(fx), var(fx))

function AbstractGPs.rand(rng::AbstractRNG, fx::FiniteBLR, samples::Int)
    X = x_as_colvecs(fx).X
    w = fx.f.mw .+ _cholesky(fx.f.Λw).U \ randn(rng, size(X, 1), samples)
    return X' * w .+ _cholesky(fx.Σy).U' * randn(rng, size(X, 2), samples)
end

function AbstractGPs.logpdf(fx::FiniteBLR, y::AbstractVector{<:Real})
    _, Bt, δy, logpdf_δy, Λεy = __compute_inference_quantities(fx, y)
    return -(logdet(Λεy) - sum(abs2, Λεy.U' \ (Bt'δy))) / 2 + logpdf_δy
end

function AbstractGPs.posterior(fx::FiniteBLR, y::AbstractVector{<:Real})
    Uw, Bt, δy, _, Λεy = __compute_inference_quantities(fx, y)

    # Compute posterior over decorrelated weights.
    mεy = Λεy \ (Bt'δy)

    # Compute posterior over weights.
    T = Λεy.U * Uw
    return BayesianLinearRegressor(fx.f.mw + Uw \ mεy, Symmetric(T'T))
end

# Computation utilised in both `logpdf` and `posterior`.
function __compute_inference_quantities(fx::FiniteBLR, y::AbstractVector{<:Real})
    X = x_as_colvecs(fx).X
    length(y) == size(X, 2) || throw(error("length(y) != size(fx.x.X, 2)"))
    blr = fx.f
    N = length(y)

    Uw = _cholesky(blr.Λw).U
    Σy = _cholesky(fx.Σy)

    Bt = Σy.U' \ (Uw' \ X)'
    δy = Σy.U' \ (y - mean(fx))

    logpdf_δy = -(N * log(2π) + logdet(Σy) + sum(abs2, δy)) / 2

    Λεy = _cholesky(Symmetric(Bt'Bt + I))

    return Uw, Bt, δy, logpdf_δy, Λεy
end

# Random function sample generation
# Following the Random API: https://docs.julialang.org/en/v1/stdlib/Random/#Hooking-into-the-Random-API
struct BLRFunctionSample{Tw<:AbstractVector}
    w::Tw
end

(s::BLRFunctionSample)(X::AbstractMatrix{<:Real}) = X's.w
(s::BLRFunctionSample)(X::ColVecs) = X.X's.w
(s::BLRFunctionSample)(X::RowVecs) = X.X * s.w

function Random.Sampler(
    ::Type{<:AbstractRNG}, blr::BayesianLinearRegressor, ::Random.Repetition
)
    return blr
end

function Random.rand(rng::AbstractRNG, blr::BayesianLinearRegressor)
    w = blr.mw .+ _cholesky(blr.Λw).U \ randn(rng, size(blr.mw))
    return BLRFunctionSample(w)
end

function Random.rand(rng::AbstractRNG, blr::BayesianLinearRegressor, dims::Dims)
    ws = blr.mw .+ _cholesky(blr.Λw).U \ randn(rng, (only(size(blr.mw)), prod(dims)))
    bs = [BLRFunctionSample(w) for w in eachcol(ws)]
    return reshape(bs, dims)
end

function Random.rand!(
    rng::AbstractRNG, A::AbstractArray{<:BLRFunctionSample}, blr::BayesianLinearRegressor
)
    ws = blr.mw .+ _cholesky(blr.Λw).U \ randn(rng, (only(size(blr.mw)), prod(size(A))))
    for i in LinearIndices(A)
        @inbounds A[i] = BLRFunctionSample(ws[:, i])
    end
    return A
end
