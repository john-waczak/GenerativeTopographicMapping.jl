module GenerativeTopographicMapping



import MLJModelInterface
import MLJModelInterface: Continuous, Multiclass, metadata_model, metadata_pkg
import Random.GLOBAL_RNG

include("GTM.jl")


## CONSTANTS
const MMI = MLJModelInterface
const PKG = "SelfOrganizingMaps"

MMI.@mlj_model mutable struct GTM <: MMI.Unsupervised
    k::Int = 16::(_ ≥ 5)
    m::Int = 4::(_ ≥ 2)
    σ::Float64 = 0.3::(_ > 0.0)
    α::Float64 = 0.1::(_ ≥ 0.0)
    tol::Float64 = 0.0001::(_ > 0.0)
    niter::Int = 200::(_ ≥ 1)
    nrepeats::Int = 4::(_ ≥ 1)
end



function MMI.fit(g::GTM, verbosity::Int, X)
    Dataset = MMI.matrix(X)

    if verbosity > 0
        printiters = true
    else
        printiters = false
    end

    # 1. build the GTM
    gtm = GenerativeTopographicMap(g.k,
                                   g.m,
                                   g.σ,
                                   Dataset;
                                   α=g.α,
                                   tol=g.tol,
                                   niter=g.niter,
                                   nrepeats=g.nrepeats,
                                   verbosity=printiters
                                   )
    # 2. Fit the GTM
    fit!(gtm, Dataset)

    # 3. collect results
    cache  = nothing
    # put class labels in the report
    classes = getModeidx(gtm, Dataset)
    report = (; :classes=>classes)

    return gtm, cache, report
end




end


