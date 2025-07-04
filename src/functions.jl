using HomotopyContinuation, LinearAlgebra, DifferentialEquations

const HC = HomotopyContinuation
const DE = DifferentialEquations


include("pseudo_witness_sets.jl")
include("gradient_hessian.jl")
include("routing_points.jl")
include("system.jl")
include("homotopy.jl")