using Documenter, BayesianLinearRegressors

makedocs(;
    modules=[BayesianLinearRegressors],
    format=Documenter.HTML(),
    pages=["Home" => "index.md"],
    repo="https://github.com/JuliaGaussianProcesses/BayesianLinearRegressors.jl/blob/{commit}{path}#L{line}",
    sitename="BayesianLinearRegressors.jl",
    authors="Will Tebbutt <wt0881@my.bristol.ac.uk>",
    assets=String[],
)

deploydocs(; repo="github.com/JuliaGaussianProcesses/BayesianLinearRegressors.jl")
