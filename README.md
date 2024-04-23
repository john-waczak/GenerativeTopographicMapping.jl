# Generative Topographic Mapping

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://john-waczak.github.io/GenerativeTopographicMapping.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://john-waczak.github.io/GenerativeTopographicMapping.jl/dev/)
[![Build Status](https://github.com/john-waczak/GenerativeTopographicMapping.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/john-waczak/GenerativeTopographicMapping.jl/actions/workflows/CI.yml?query=branch%3Amain)

A Julia package for Generative Topographic Mapping originally introduced by [Bishop and Svensen](https://www.microsoft.com/en-us/research/wp-content/uploads/1998/01/bishop-gtm-ncomp-98.pdf) with inspiration from the [UGTM](https://ugtm.readthedocs.io/en/latest/overview.html) python package. This implementation provides wrappers for use withing the MLJ framework.


# Using the package

To train a GTM model first load MLJ and this package.
```julia
using MLJ, GenerativeTopographicMapping
```

The GTM can then be instantiated in the usual way for an unsupervised method:

```julia
gtm = GTM()
mach = machine(gtm, df)
fit!(mach)
```

Calling `fitted_params` on the trained machine will return a tuple holding the underlying `gtm` struct if further inspection is desired.

The GTM model learns a transformation from a latent space of $k\times k$ nodes. The result is a set of responsibilities for each latent node. By computing the mean of this distribution for each record, we can use the GTM as a nonlinear dimensionality reduction scheme

```julia
means = MLJ.transform(mach, X)
```

Alternatively, the GTM can be viewed as performing an unsupervised classification into $k\times k$ classes. The class label for each record can be obtained as the index of the node corresponding to the node of the responsibility distribution via

```julia
class_label = MLJ.predict(mach, X)
```

If the full responsibility distribution is desired, you can `predict_responsability` on the trained machine

```julia
R = predict_responsibility(mach, X)
```

