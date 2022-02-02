struct BasisFunctionBayesianLinearRegressor{Tblr<:BayesianLinearRegressor,Tϕ} <: AbstractGP
    blr::Tblr
    ϕ::Tϕ
end

function (bfblr::BasisFunctionBayesianLinearRegressor)(x::AbstractVector, args...)
    return bfblr.blr(bfblr.ϕ(x), args...)
end
