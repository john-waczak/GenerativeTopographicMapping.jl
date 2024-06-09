include("../src/GenerativeTopographicMapping.jl")

using CairoMakie
using MLJ
using DataFrames
using TernaryDiagrams
using Distributions
using StableRNGs





rng = StableRNG(42)

include("./utils/makie-defaults.jl")
set_theme!(mints_theme)
update_theme!(
    figure_padding=30,
    Axis=(
        xticklabelsize=20,
        yticklabelsize=20,
        xlabelsize=22,
        ylabelsize=22,
        titlesize=25,
    ),
    Colorbar=(
        ticklabelsize=20,
        labelsize=22
    )
)


k = 25
m = 5
s = 1
λ = 0.1
Nᵥ = 3


# visualize the prior distribution
αs = [0.1, 0.5, 0.75, 1.0, 5.0]
for α ∈ αs
    α = α * ones(Nᵥ)

    Ξ = GenerativeTopographicMapping.get_barycentric_grid_coords(k, Nᵥ)
    πk = zeros(size(Ξ, 2))
    f_dirichlet = Dirichlet(α)

    e = 0.5 * (1/k)  # offset to deal with Inf value on boundary

    for j ∈ axes(Ξ,2)
        p = Ξ[:,j]
        for i ∈ axes(Ξ,1)
            if Ξ[i,j] == 0.0
                p[i] = e
            end
        end
        p = p ./ sum(p)
        πk[j] = pdf(f_dirichlet, p)
    end

    πk = πk ./ sum(πk)

    fig = Figure();
    ax = Axis(fig[1, 1], aspect=1, title="p(ξₖ) with α=$(α)", titlefont=:regular, titlealign=:left);

    ternaryaxis!(
        ax,
        tick_fontsize=15,
    );

    ts = ternaryscatter!(
        ax,
        Ξ[1,:],
        Ξ[2,:],
        Ξ[3,:],
        # color = log.(πk),
        color = πk,
        marker = :circle,
        markersize = 18,
    )

    xlims!(ax, -0.2, 1.2)
    ylims!(ax, -0.3, 1.1)
    hidedecorations!(ax)
    hidespines!(ax)
    text!(ax, Point2f(0.1, 0.5), text="ξ₃", fontsize=22)
    text!(ax, Point2f(0.825, 0.5), text="ξ₂", fontsize=22)
    text!(ax, Point2f(0.5, -0.175), text="ξ₁", fontsize=22)

    cb = Colorbar(fig[1,2], ts, label="πₖ")

    fig

    save("./figures/prior_$(α[1]).png", fig)
    save("./figures/prior_$(α[1]).pdf", fig)

end








# Test 1: Linear Mixing
λs = range(350, stop=750, length=1000)
# λs = range(400, stop=700, length=1000)

Rb = exp.(-(λs .- 460.0).^2 ./(2*(22)^2))
Rg = exp.(-(λs .- 525.0).^2 ./(2*(28)^2))
Rr = exp.(-(λs .- 625.0).^2 ./(2*(20)^2))

# Rb = exp.(-(abs.(λs .- 450.0)./ 10).^1)
# Rg = exp.(-(abs.(λs .- 530.0)./ 30).^2)
# Rr = exp.(-(abs.(λs .- 620.0)./ 40).^10)

# Rb = exp.(-(abs.(λs .- 476.0)./ 10).^1)
# Rg = exp.(-(abs.(λs .- 530.0)./ 30).^2)
# Rr = exp.(-(abs.(λs .- 605.0)./ 40).^10)

Rb .= Rb./maximum(Rb)
Rg .= Rg./maximum(Rg)
Rr .= Rr./maximum(Rr)

fig = Figure();
ax = Axis(fig[2,1], xlabel="λ", ylabel="Reflectance");
lr = lines!(ax, λs, Rr, color=mints_colors[2], linewidth=2)
lg = lines!(ax, λs, Rg, color=mints_colors[1], linewidth=2)
lb = lines!(ax, λs, Rb, color=mints_colors[3], linewidth=2)
fig[1,1] = Legend(fig, [lr, lg, lb], ["Red Band", "Green Band", "Blue Band"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=13, height=-5)
fig

save("./figures/rbg-orig.png", fig)
save("./figures/rbg-orig.pdf", fig)



# generate dataframe of reflectance values from combined dataset

# Npoints = 10_000
Npoints = 25_000
α_true = 0.05 * ones(3)
f_dir = Dirichlet(α_true)
abund = rand(rng, f_dir, Npoints)



# visualize the distribution of abundances
fig = Figure();
ax = Axis(fig[1, 1], aspect=1);

ternaryaxis!(
    ax,
    hide_triangle_labels=false,
    hide_vertex_labels=false,
    labelx_arrow = "Red",
    label_fontsize=20,
    tick_fontsize=15,
)

ternaryscatter!(
    ax,
    abund[1,:],
    abund[2,:],
    abund[3,:],
    color=[CairoMakie.RGBf(abund[:,i]...) for i ∈ 1:Npoints],
    marker=:circle,
    markersize = 10,
)

# the triangle is drawn from (0,0) to (0.5, sqrt(3)/2) to (1,0).
xlims!(ax, -0.2, 1.2) # to center the triangle and allow space for the labels
ylims!(ax, -0.3, 1.1)
hidedecorations!(ax) # to hide the axis decorations

text!(ax, Point2f(0.1, 0.5), text="Blue", fontsize=22)
text!(ax, Point2f(0.825, 0.5), text="Green", fontsize=22)
text!(ax, Point2f(0.5, -0.175), text="Red", fontsize=22)

fig

save("./figures/abundance-orig.png", fig)
save("./figures/abundance-orig.pdf", fig)



X = zeros(Npoints, length(λs));    # linear mixing
Xnoise = zeros(Npoints, length(λs));
X2 = zeros(Npoints, length(λs));   # non-linear mixing
for i ∈ axes(X, 1)
    X[i,:] .= (abund[1,i] .* Rr) .+ (abund[2,i] .* Rg) .+ (abund[3,i] .* Rb)
    Xnoise[i,:] .= X[i,:] .+ 0.01*(2 .* rand(rng, length(λs)) .- 0.5)


    X2[i,:] .= (abund[1,i] .* Rr) .+ (abund[2,i] .* Rg) .+ (abund[3,i] .* Rb) .+ (abund[1,i] * abund[2,i] .* Rr .* Rg)  .+ (abund[1,i] * abund[2,i] .* Rr .* Rg) .+ (abund[1,i] * abund[3,i] .* Rr .* Rb)  .+ (abund[2,i] * abund[3,i] .* Rg .* Rb)

    # X[i,:] .= X[i,:] ./ maximum(X[i,:])
    # X2[i,:] .= X2[i,:] ./ maximum(X2[i,:])
end


names = Symbol.(["λ_" * lpad(i, 3, "0") for i ∈ 1:length(λs)])
X = DataFrame(X, names);
X2 = DataFrame(X2, names);

fig = Figure();
ax = Axis(fig[1,1], xlabel="λ", ylabel="Reflectance")
for i ∈ 1:10:nrow(X)
    Rᵢ = Array(X[i,:])
    lines!(ax, λs, Rᵢ, color=CairoMakie.RGBAf(abund[:,i]..., 0.25), linewidth=1)
end
fig

save("./figures/sample-spectra.png", fig)
save("./figures/sample-spectra.pdf", fig)


fig = Figure();
ax = Axis(fig[1,1], xlabel="λ", ylabel="Reflectance")
for i ∈ 1:10:nrow(X2)
    Rᵢ = Array(X2[i,:])
    lines!(ax, λs, Rᵢ, color=CairoMakie.RGBAf(abund[:,i]..., 0.25), linewidth=1)
end
fig

save("./figures/sample-spectra-nonlinear.png", fig)
save("./figures/sample-spectra-nonlinear.pdf", fig)



# ---------------------------------------------
# ------------- LINEAR MODEL -----------------
# ---------------------------------------------

if !ispath("./figures/linear")
    mkpath("./figures/linear")
end

k = 50
m = 10
s = 1
λ = 0.001
Nᵥ = 3

# gsm_l = GenerativeTopographicMapping.GSM(k=k, m=m, Nv=Nᵥ, s=s, λ=λ,  nonlinear=false, linear=true, bias=false, make_positive=true, tol=1e-9, nepochs=100, rand_init=true, rng=StableRNG(42))
gsm_l = GenerativeTopographicMapping.GSM(k=k, m=m, Nv=Nᵥ, s=s, λ=λ,  nonlinear=false, linear=true, bias=false, make_positive=true, tol=1e-9, nepochs=100, rand_init=false, rng=StableRNG(42))
mach_l = machine(gsm_l, X)
fit!(mach_l, verbosity=1)
abund_l = DataFrame(MLJ.transform(mach_l, X));

model = fitted_params(mach_l)[:gsm]


rpt = report(mach_l)
node_means = rpt[:node_data_means]
llhs = rpt[:llhs]
idx_vertices = rpt[:idx_vertices]
std = sqrt(rpt[:β⁻¹])


model = fitted_params(mach_l)[:gsm]
πk = model.πk
Ξ = model.Ξ

# plot log likelihoods
fig = Figure();
ax = Axis(fig[1,1], xlabel="Iteration", ylabel="Log-likelihood", yscale=log10)
lines!(ax, 2:length(llhs), llhs[2:end], linewidth=3)
fig

save("./figures/linear/lllh-linear-a=5.png",fig)
save("./figures/linear/lllh-linear-a=5.pdf",fig)


# plot distribution linear data
fig = Figure();
ax = Axis(fig[1, 1], aspect=1);

ternaryaxis!(
    ax,
    hide_triangle_labels=false,
    hide_vertex_labels=false,
    labelx_arrow = "Red",
    label_fontsize=20,
    tick_fontsize=15,
)

ternaryscatter!(
    ax,
    abund_l[:,1],
    abund_l[:,2],
    abund_l[:,3],
    color=[CairoMakie.RGBf(abund_l[i,:]...) for i ∈ 1:Npoints],
    marker=:circle,
    markersize = 10,
)

# the triangle is drawn from (0,0) to (0.5, sqrt(3)/2) to (1,0).
xlims!(ax, -0.2, 1.2) # to center the triangle and allow space for the labels
ylims!(ax, -0.3, 1.1)
hidedecorations!(ax) # to hide the axis decorations
fig

save("./figures/linear/abundance-fit.png", fig)
save("./figures/linear/abundance-fit.pdf", fig)


# plot endmembers for linear data
fig = Figure();
ax = Axis(fig[2,1], xlabel="λ", ylabel="Reflectance");

lr = lines!(ax, λs, Rr, color=mints_colors[2], linewidth=3)
lg = lines!(ax, λs, Rg, color=mints_colors[1], linewidth=3)
lb = lines!(ax, λs, Rb, color=mints_colors[3], linewidth=3)

ls_fit = []
linestyles = [:solid, :dash, :dot]
i = 1
for idx ∈ idx_vertices
    Rout = node_means[:,idx]
    # Rout .= Rout ./ maximum(Rout)
    band!(ax, λs, Rout .- (2*std), Rout .+ (2*std), color=(:gray, 0.5))
    li = lines!(ax, λs, Rout, color=:gray, linewidth=2, linestyle=linestyles[i])
    push!(ls_fit, li)
    i += 1
end

fig[1,1] = Legend(fig, [lr, lg, lb, ls_fit...], ["Red Band", "Green Band", "Blue Band", "Vertex 1", "Vertex 2", "Vertex 3"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=13, height=-5)

fig


save("./figures/linear/endmembers-extracted.png", fig)
save("./figures/linear/endmembers-extracted.pdf", fig)



# ---------------------------------------------
# ------------- LINEAR NOISY MODEL-----------------
# ---------------------------------------------

if !ispath("./figures/linear-noisy")
    mkpath("./figures/linear-noisy")
end



k = 50
m = 10
s = 1
λ = 0.01
Nᵥ = 3

# α = ones(Nᵥ)
α = α_true

gsm_l = GenerativeTopographicMapping.GSM(k=k, m=m, Nv=Nᵥ, s=s, λ=λ,  α=α, linear_only=true, make_positive=true, tol=1e-9, nepochs=100, rand_init=true, rng=StableRNG(42))
mach_l = machine(gsm_l, Xnoise)
fit!(mach_l, verbosity=1)
abund_l = DataFrame(MLJ.transform(mach_l, Xnoise));

rpt = report(mach_l)
node_means = rpt[:node_data_means]
llhs = rpt[:llhs]
idx_vertices = rpt[:idx_vertices]
std = sqrt(rpt[:β⁻¹])

# plot log likelihoods
fig = Figure();
ax = Axis(fig[1,1], xlabel="Iteration", ylabel="Log-likelihood", yscale=log10)
lines!(ax, 3:length(llhs), llhs[3:end], linewidth=3)
fig
save("./figures/linear-noisy/lllh-linear.png",fig)
save("./figures/linear-noisy/lllh-linear.pdf",fig)


# plot distribution linear data
fig = Figure();
ax = Axis(fig[1, 1], aspect=1);

ternaryaxis!(
    ax,
    hide_triangle_labels=false,
    hide_vertex_labels=false,
    labelx_arrow = "Red",
    label_fontsize=20,
    tick_fontsize=15,
)

ternaryscatter!(
    ax,
    abund_l[:,1],
    abund_l[:,2],
    abund_l[:,3],
    color=[CairoMakie.RGBf(abund_l[i,:]...) for i ∈ 1:Npoints],
    marker=:circle,
    markersize = 10,
)

# the triangle is drawn from (0,0) to (0.5, sqrt(3)/2) to (1,0).
xlims!(ax, -0.2, 1.2) # to center the triangle and allow space for the labels
ylims!(ax, -0.3, 1.1)
hidedecorations!(ax) # to hide the axis decorations
fig

save("./figures/linear-noisy/abundance-fit.png", fig)
save("./figures/linear-noisy/abundance-fit.pdf", fig)


# plot endmembers for linear data
fig = Figure();
ax = Axis(fig[2,1], xlabel="λ", ylabel="Reflectance");

lr = lines!(ax, λs, Rr, color=mints_colors[2], linewidth=3)
lg = lines!(ax, λs, Rg, color=mints_colors[1], linewidth=3)
lb = lines!(ax, λs, Rb, color=mints_colors[3], linewidth=3)

ls_fit = []
linestyles = [:solid, :dash, :dot]
i = 1
for idx ∈ idx_vertices
    Rout = node_means[:,idx]
    # Rout .= Rout ./ maximum(Rout)
    band!(ax, λs, Rout .- (2*std), Rout .+ (2*std), color=(:gray, 0.5))
    li = lines!(ax, λs, Rout, color=:gray, linewidth=2, linestyle=linestyles[i])
    push!(ls_fit, li)
    i += 1
end

fig[1,1] = Legend(fig, [lr, lg, lb, ls_fit...], ["Red Band", "Green Band", "Blue Band", "Vertex 1", "Vertex 2", "Vertex 3"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=13, height=-5)

fig

save("./figures/linear-noisy/endmembers-extracted.png", fig)
save("./figures/linear-noisy/endmembers-extracted.pdf", fig)







# # fit non-linear mixing model
# gsm_l = GenerativeTopographicMapping.GSMLog(k=k, m=m, Nv=Nᵥ, s=s, λ=λ,  α=α, tol=1e-9, nepochs=100, rand_init=true, rng=StableRNG(42))

# gsm_nl = GenerativeTopographicMapping.GSM(k=k, m=m, Nv=Nᵥ, s=s, λ=λ, α=α, η=0.0001,  tol=1e-5, nepochs=100, rng=StableRNG(42))
# mach_nl = machine(gsm_nl, X2)
# fit!(mach_nl, verbosity=1)
# abund_nl = DataFrame(MLJ.transform(mach_nl, X2));

# # plot distribution non-linear data
# fig = Figure();
# ax = Axis(fig[1, 1], aspect=1);

# ternaryaxis!(
#     ax,
#     hide_triangle_labels=false,
#     hide_vertex_labels=false,
#     labelx_arrow = "Red",
#     label_fontsize=20,
#     tick_fontsize=15,
# )

# ternaryscatter!(
#     ax,
#     abund_nl[:,1],
#     abund_nl[:,2],
#     abund_nl[:,3],
#     color=[CairoMakie.RGBf(abund_nl[i,:]...) for i ∈ 1:Npoints],
#     marker=:circle,
#     markersize = 10,
# )

# # the triangle is drawn from (0,0) to (0.5, sqrt(3)/2) to (1,0).
# xlims!(ax, -0.2, 1.2) # to center the triangle and allow space for the labels
# ylims!(ax, -0.3, 1.1)
# hidedecorations!(ax) # to hide the axis decorations
# fig

# save("./notes/figures/abundance-fit-nonlinear.png", fig)
# save("./notes/figures/abundance-fit-nonlinear.pdf", fig)





# # plot endmembers for non-linear dataset
# gsm_mdl = fitted_params(mach_nl)[:gsm]
# rpt = report(mach_nl
# M = gsm_mdl.M                          # RBF centers
# Ξ = rpt[:Ξ]                            # Latent Points
# Ψ = rpt[:W] * rpt[:Φ]'                 # Projected Node Means
# llhs = rpt[:llhs]
# idx_vertices = rpt[:idx_vertices]


# fig = Figure();
# ax = Axis(fig[2,1], xlabel="λ", ylabel="Reflectance");
# lr = lines!(ax, λs, Rr, color=mints_colors[2], linewidth=2)
# lg = lines!(ax, λs, Rg, color=mints_colors[1], linewidth=2)
# lb = lines!(ax, λs, Rb, color=mints_colors[3], linewidth=2)

# ls_fit = []
# linestyles = [:solid, :dash, :dot]
# i = 1
# for idx ∈ idx_vertices
#     #lines!(ax, λs, exp.(Ψ[:, idx] .+ (gsm_mdl.β⁻¹/2)), color=:gray)
#     Rout = exp.(Ψ[:, idx])
#     Rout .= Rout ./ maximum(Rout)
#     li = lines!(ax, λs, Rout, color=:gray, linewidth=3, linestyle=linestyles[i])
#     push!(ls_fit, li)
#     i += 1
# end

# fig[1,1] = Legend(fig, [lr, lg, lb, ls_fit...], ["Red Band", "Green Band", "Blue Band", "Vertex 1", "Vertex 2", "Vertex 3"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=13, height=-5)

# fig

# save("./notes/figures/endmembers-extracted-nonlinear.png", fig)
# save("./notes/figures/endmembers-extracted-nonlinear.pdf", fig)





# # is there a prior we can add to the Weights to force them to be positive
# # additionally, we may want to explore a prior which encourages sparsity in the data space...


# # Need to add function to GTM to provide vertex indices in fit report, we can get them using with_replacement_combinations([1,N], k)
# # Fix PCA to guarantee number of vectors



