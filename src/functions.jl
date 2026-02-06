using HomotopyContinuation, LinearAlgebra, DifferentialEquations, LightGraphs, ProgressMeter

const HC = HomotopyContinuation
const DE = DifferentialEquations

import HomotopyContinuation.evaluate!
import HomotopyContinuation.evaluate_and_jacobian!
import HomotopyContinuation.evaluate
import HomotopyContinuation.taylor!
import HomotopyContinuation.ModelKit.evaluate
import HomotopyContinuation.ModelKit.nvariables
import HomotopyContinuation.ModelKit.variables

include("pseudo_witness_sets.jl")
include("gradient_cache.jl")
include("hypersurfaces.jl")
include("routing_functions.jl")
include("homotopy.jl")
include("ode_solving.jl")
include("graph.jl")
