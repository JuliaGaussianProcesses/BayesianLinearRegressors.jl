struct BasisFunctionRegressor{Tblr<:BayesianLinearRegressor,Tϕ} <: AbstractGP
    blr::Tblr
    ϕ::Tϕ
end

function (bfblr::BasisFunctionRegressor)(x::AbstractVector, args...)
    return bfblr.blr(bfblr.ϕ(x), args...)
end
