# Random function sample generation
# Following the Random API: https://docs.julialang.org/en/v1/stdlib/Random/#Hooking-into-the-Random-API
"""
    BLRFunctionSample{Tw,Tϕ}

A function sampled from
 `b::Union{BayesianLinearRegressor,BasisFunctionRegressor}` by taking a fixed
 weight sample `w ~ p(w)`. `ϕ` is the feature mapping if sampled from a
 `BasisFunctionRegressor`, or is the identity function if sampled from a
 `BayesianLinearRegressor`.
"""
struct BLRFunctionSample{Tw<:AbstractVector,Tϕ}
    w::Tw
    ϕ::Tϕ
end

(s::BLRFunctionSample)(X::AbstractMatrix{<:Real}) = s.ϕ(X)'s.w
(s::BLRFunctionSample)(X::ColVecs) = s.ϕ(X).X's.w
(s::BLRFunctionSample)(X::RowVecs) = s.ϕ(X).X * s.w

BLRorBasisFunction = Union{BayesianLinearRegressor,BasisFunctionRegressor}

function Random.Sampler(::Type{<:AbstractRNG}, blr::BLRorBasisFunction, ::Random.Repetition)
    return blr
end

function Random.rand(rng::AbstractRNG, b::BLRorBasisFunction)
    blr, ϕ = _blr_and_mapping(b)
    w = blr.mw .+ _cholesky(blr.Λw).U \ randn(rng, size(blr.mw))
    return BLRFunctionSample(w, ϕ)
end

function Random.rand(rng::AbstractRNG, b::BLRorBasisFunction, dims::Dims)
    blr, ϕ = _blr_and_mapping(b)
    ws = blr.mw .+ _cholesky(blr.Λw).U \ randn(rng, (only(size(blr.mw)), prod(dims)))
    bs = [BLRFunctionSample(w, ϕ) for w in eachcol(ws)]
    return reshape(bs, dims)
end

function Random.rand!(
    rng::AbstractRNG, A::AbstractArray{<:BLRFunctionSample}, b::BLRorBasisFunction
)
    blr, ϕ = _blr_and_mapping(b)
    ws = blr.mw .+ _cholesky(blr.Λw).U \ randn(rng, (only(size(blr.mw)), prod(size(A))))
    for i in LinearIndices(A)
        @inbounds A[i] = BLRFunctionSample(ws[:, i], ϕ)
    end
    return A
end

_blr_and_mapping(b::BayesianLinearRegressor) = b, identity
_blr_and_mapping(b::BasisFunctionRegressor) = b.blr, b.ϕ
