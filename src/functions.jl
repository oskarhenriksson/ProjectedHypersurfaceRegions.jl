using HomotopyContinuation, LinearAlgebra

const HC = HomotopyContinuation
const DE = DifferentialEquations

include("pseudo_witness_sets.jl")
include("gradient_hessian.jl")
include("routing_pts.jl")