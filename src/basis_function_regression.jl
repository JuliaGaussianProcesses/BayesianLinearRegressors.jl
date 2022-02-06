"""
    BasisFunctionRegressor{Tblr,Tϕ}

A Basis Function Regressor represents a Bayesian Linear Regressor where the
input `x` is first mapped to a feature space through a basis function ϕ.

ϕ must be a function which accepts one of the allowed input types for
BayesianLinearRegressors (ColVecs, RowVecs or Matrix{<:Real} - see the package
readme for more details) and it must output one of these allowed types.

```jldoctest; output = false
x = RowVecs(hcat(range(-1.0, 1.0, length=5)))
blr = BayesianLinearRegressor(zeros(2), Diagonal(ones(2)))

ϕ(x::RowVecs) = RowVecs(hcat(ones(length(x)), prod.(x)))
bfr = BasisFunctionRegressor(blr, ϕ)

var(bfr(x))

# output

5-element Vector{Float64}:
 2.0
 1.25
 1.0
 1.25
 2.0
```

See [1], Section 3.1 for more details on basis function regression.

[1] - C. M. Bishop. "Pattern Recognition and Machine Learning". Springer, 2006.
"""
struct BasisFunctionRegressor{Tblr<:BayesianLinearRegressor,Tϕ} <: AbstractGP
    blr::Tblr
    ϕ::Tϕ
end

function (bfblr::BasisFunctionRegressor)(x::AbstractVector, args...)
    return bfblr.blr(bfblr.ϕ(x), args...)
end
