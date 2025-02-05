# ITensor deconstructed: rewriting ITensors.jl

This repository contains a [Pluto.jl](https://github.com/fonsp/Pluto.jl) notebook
[itensor-deconstructed.jl](https://github.com/ITensor/itensor-deconstructed/blob/main/itensor-deconstructed.jl)
that demonstrates a new set of packages that are being developed as part of a
[rewrite of ITensors.jl](https://github.com/ITensor/ITensors.jl/pull/1611).

To run the code, clone the repository and enter the directory where it is cloned:
```
git clone https://github.com/ITensor/itensor-deconstructed.git
cd itensor-deconstructed
```
Then, install and launch Pluto.jl:
```julia
julia> import Pkg; Pkg.add("Pluto")

julia> Pluto.run()
```
Pluto will open in the browser, where you can open the file `itensor-deconstructed.jl` that you can find
in your cloned version of this repository.

<!---
You can view a static webpage generated from the notebook here:
[https://itensor.github.io/itensor-deconstructed/itensor-deconstructed.html](https://itensor.github.io/itensor-deconstructed/itensor-deconstructed.html).
--->

Note that the notebook relies on a number of packages that are registered in the
[ITensor registry](https://github.com/ITensor/ITensorRegistry), you will need to add that
registry before you can run the notebook locally. See the instructions
[here](https://github.com/ITensor/ITensorRegistry?tab=readme-ov-file#using-the-registry).

<!---
The static webpage generation workflow is based on
[this template](https://github.com/JuliaPluto/static-export-template), with the addition of
[this line](https://github.com/ITensor/itensor-deconstructed/blob/97c67766fd8fbd3d4b23c0052ca042c9ce3976b7/.github/workflows/ExportPluto.yaml#L49)
to add the ITensor registry so the dependencies can be installed properly.
--->
