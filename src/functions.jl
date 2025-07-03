using HomotopyContinuation, LinearAlgebra, DifferentialEquations

const HC = HomotopyContinuation
const DE = DifferentialEquations


include("gradient_hessian.jl")
include("pseudo_witness_sets.jl")
include("routing_points.jl")
