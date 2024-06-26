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

# Define the basis elements
linear_elem(Δ, m) = (Δ ≤ 1/m) ? m*((1/m)-Δ) : 0.0
quadratic_elem(Δ, m) = (Δ ≤ 1/m) ? m^2*((1/m)-Δ)*(Δ+(1/m)) : 0.0


linear_elem_big(Δ, s) = (Δ ≤ s) ? (s-Δ)/s : 0.0
quadratic_elem_big(Δ, s) = (Δ ≤ s) ? (s-Δ)*(Δ+s)/s^2 : 0.0
