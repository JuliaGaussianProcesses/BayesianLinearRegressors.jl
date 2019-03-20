module BLR

using Random, LinearAlgebra, Distributions

import Random: rand
import Distributions: logpdf

const AV = AbstractVector
const AM = AbstractMatrix

include("blr.jl")

end # module
