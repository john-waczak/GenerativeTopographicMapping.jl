using Combinatorics
using LinearAlgebra

function get_barycentric_grid_coords(Nₑ, Nᵥ)
    D = Nᵥ - 1
    Npts = binomial(Nₑ+D-1, D)

    rᵥ = collect(with_replacement_combinations([1,Nₑ], D))
    rᵢ = collect(with_replacement_combinations(1:Nₑ, D))

    Tinv = inv(hcat(rᵥ[1:end-1]...) .- rᵥ[end])
    R = hcat(rᵢ...) .- rᵥ[end]

    Λ  = zeros(Nᵥ, Npts)
    Λ[1:end-1, :] .= Tinv*R
    Λ[end:end,:] .= 1 .- sum(Λ[1:end-1, :], dims=1)

    return Λ
end
