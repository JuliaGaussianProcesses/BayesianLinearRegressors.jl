module BLR

using Random, LinearAlgebra, Distributions, Statistics

import Statistics: mean, cov
import Random: rand
import Distributions: logpdf

const AV = AbstractVector
const AM = AbstractMatrix

include("blr.jl")

end # module
