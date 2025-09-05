using HomotopyContinuation, LinearAlgebra, DifferentialEquations

const HC = HomotopyContinuation
const DE = DifferentialEquations


include("pseudo_witness_sets.jl")
include("gradient_hessian.jl")
include("routing_points.jl")
include("slice_system.jl") # For now, I will rename system to slice_system until it works properly.
include("homotopy.jl")