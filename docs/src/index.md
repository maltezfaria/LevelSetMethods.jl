```@meta
CurrentModule = LevelSetMethods
```

# LevelSetMethods

Documentation for [LevelSetMethods](https://github.com/maltezfaria/LevelSetMethods.jl).

## Installation

LevelSetMethods.jl is not yet registered in the Julia package registry. To install it, you can use the following command:

```julia
using Pkg; Pkg.add("https://github.com/maltezfaria/LevelSetMethods.jl")
```

## Overview

This package provides a set of tools to solve level set equations in Julia. The main features are:
- ...

## Basic usage

Solving a level-set equation using [LevelSetMethods](https://github.com/maltezfaria/LevelSetMethods.jl) consists of the following steps:

1. Create a grid
2. Initialize the level set function
3. Chose the [`LevelSetTerm`](@ref)s to be used
4. Pick a [`TimeIntegrator`](@ref)
5. Create a [`LevelSetEquation`](@ref)
6. Step the equation in time using [`integrate!`](@ref)

...

For examples, see the examples section.

## Contributors

```@raw html
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
```
