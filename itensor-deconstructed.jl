### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 6cbb26d9-10c7-4f3b-bfc6-03dc5fb53a34
using Kroki

# ╔═╡ 19579878-a2fc-46b6-9a20-b3af72ae2464
using SparseArraysBase: SparseArrayDOK

# ╔═╡ 327ef53d-3fd9-4740-b670-b7b29a6a5adb
using SparseArraysBase: eachstoredindex, isstored

# ╔═╡ 19b1594c-4260-4c7c-91f1-eae0013b51e0
using TensorAlgebra: contract

# ╔═╡ a042962b-0a5b-426d-83d1-68daa7518ff1
using NamedDimsArrays: NamedDimsArray

# ╔═╡ ca1cc56f-3400-43eb-811f-40bcd49a8438
using NamedDimsArrays: aligndims, dimnames, unname

# ╔═╡ 5e2dd3f1-de9f-416d-b259-6658f2781483
using BlockSparseArrays: BlockSparseArray

# ╔═╡ 7edf8314-0645-4e52-afc6-f9ac3fe8de9a
using BlockArrays: blocks

# ╔═╡ 8cffacf7-b87c-4a8a-9c2f-edeeb791bbd5
using BlockArrays: Block

# ╔═╡ 1e35e904-fc55-45df-b7cd-4996744fce16
using ITensorBase: ITensor, Index, inds

# ╔═╡ c836c0d2-1685-4a83-a436-d251c31d8c15
using NamedDimsArrays: AbstractNamedDimsArray

# ╔═╡ fb236256-490a-4524-a5f3-36ecdf927fe0
using SymmetrySectors: U1, dual

# ╔═╡ 3a90675f-1a5a-4fc2-8698-96c40394467b
using GradedUnitRanges: gradedrange

# ╔═╡ 957bab60-86a5-4cbb-84f8-e51ec0ce8723
md"""
# ITensor deconstructed
### Splitting apart ITensors.jl for better _composibility_, _generalizability_, _extensibility_, and _maintainability_

- [https://github.com/ITensor/itensor-deconstructed](https://github.com/ITensor/itensor-deconstructed)
"""

# ╔═╡ 75070daa-0985-4886-95a7-b671f6e9005f
md"""
## Current ITensors.jl design
"""

# ╔═╡ e973f9cb-c8ca-4e3c-afc2-adce2fe9f1f9
mermaid"""
graph TD
    ITensors(ITensors.jl) --> NDTensors(NDTensors.jl)
    NDTensors --> Tensor(NDTensors.Tensor)
    Tensor --> Dense(NDTensors.Dense)
    Tensor --> Diag(NDTensors.Diag)
    Tensor --> BlockSparse(NDTensors.BlockSparse)
    Tensor --> DiagBlockSparse(NDTensors.DiagBlockSparse)
    Dense --> Array(Base.Array)
    Dense --> CUDA(CUDA.jl)
    Diag --> Array
    Diag --> CUDA
    BlockSparse --> Array
    BlockSparse --> CUDA
    DiagBlockSparse --> Array
    DiagBlockSparse --> CUDA
"""

# ╔═╡ f1385767-d637-42f1-827e-9aadd1ddfed7
md"""
### Problems with the current design:
- Tensors and tensor operations are based around a single large package, **NDTensors.jl**, and a single type `NDTensors.Tensor` that has a variety of backends ("storage types").
- Flat structure makes the library monolithic, making it hard to maintain, test, extend, and compose parts:
  - Block diagonal tensors should be a composition of diagonal and block sparse, but that is hard to do with the current design, which leads to a lot of code duplication and maintenance challengs.
  - GPU support is much better now (thanks to **Karl Pierce**), but a better library design could have made it easier.
  - Challenging to generalize to ambitious new types, like non-abelian symmetries, distributed tensors, new GPU types, etc.
"""

# ╔═╡ 79fa6a5e-ffaa-46b9-97a3-390d045e543f
md"""
## New ITensors.jl design (**in progress**)
"""

# ╔═╡ ad59d756-1554-4931-a0fe-dbc4d41d9d42
mermaid"""
graph TD
    ITensors(ITensors.jl) --> NamedDimsArrays(NamedDimsArrays.jl)
    NamedDimsArrays --> Array(Base.Array)
    NamedDimsArrays --> CUDA(CUDA.jl)
    NamedDimsArrays --> SparseArraysBase(SparseArraysBase.jl)
    NamedDimsArrays --> DiagonalArrays(DiagonalArrays.jl)
    NamedDimsArrays --> BlockSparseArrays(BlockSparseArrays.jl)
    NamedDimsArrays --> FusionTensors(FusionTensors.jl)
    DiagonalArrays --> SparseArraysBase
    DiagonalArrays --> Array
    DiagonalArrays --> CUDA
    BlockSparseArrays --> SparseArraysBase
    BlockSparseArrays --> Array
    BlockSparseArrays --> CUDA
    BlockSparseArrays --> SymmetrySectors(SymmetrySectors.jl)
    FusionTensors --> BlockSparseArrays
    FusionTensors --> SymmetrySectors
"""

# ╔═╡ c604e497-24b4-499d-a66e-0e04e3f4cded
md"""
### Advantages of the new design
- Based around a series of standalone Julia packages that implement Julia arrays adhering to the `Base.AbstractArray` interface which are then composed with each other to get more sophisticated behavior.
- ITensors are now just special types of a more general **NamedDimsArrays.jl**[^1] package, making the design more flexible and generalizable. 
- Easier maintenance, testing (packages will have self-contained tests, docs, maintainence, etc.).
- Better composibility, code sharing.
  - **BlockSparseArrays.jl**[^2] and **DiagonalArrays.jl**[^3] are both based on code shared in **SparseArraysBase.jl**[^4], and can be composed with each other in various ways.
  - The new `BlockSparseArray` type is much more flexible than the current one in NDTensors.jl, for example it is designed to allow storing arbitrary kinds of data as blocks, supports many more slicing and broadcasting operations, etc.
- The new design will make it easier to enable ITensors to have new kinds of storage types:
  - Non-abelian tensors (**FusionTensors.jl**[^5], in progress by **Olivier Gauthé**, with the help of **Lukas Devos** and **Miles Stoudenmire**.)
  - GPU (in progress in the new system but a lot work already)
  - Distributed (future work)
  - Automatic fermion sign support (future work)
  - etc.

[1]: [https://github.com/ITensor/NamedDimsArrays.jl](https://github.com/ITensor/NamedDimsArrays.jl)

[2]: [https://github.com/ITensor/BlockSparseArrays.jl](https://github.com/ITensor/BlockSparseArrays.jl)

[3]: [https://github.com/ITensor/DiagonalArrays.jl](https://github.com/ITensor/DiagonalArrays.jl)

[4]: [https://github.com/ITensor/SparseArraysBase.jl](https://github.com/ITensor/SparseArraysBase.jl)

[5]: [https://github.com/ITensor/FusionTensors.jl](https://github.com/ITensor/FusionTensors.jl)
"""

# ╔═╡ 34c5e555-3919-42a1-af28-103edfeb9140
md"""
### How to install
- All still a work in progress, used internally right now.
- For now, they are available at the new ITensor registry[^1], set up by **Lukas Devos**:
```julia
import Pkg
# Add the registry (only need to do this once per machine)
Pkg.Registry.add(url="https://github.com/ITensor/ITensorRegistry")

# Add a package from the registry
Pkg.add("SparseArraysBase")

# Use the package
using SparseArraysBase
a = SparseArrayDOK{Float64}(2, 2)
```
- We plan to register them in Julia's General registry once they are ready and being used inside one of the next major ITensors.jl release.

[1]: [https://github.com/ITensor/ITensorRegistry](https://github.com/ITensor/ITensorRegistry)
"""

# ╔═╡ bc6f11e4-2fde-4a2d-9d53-2bd7d5c497fe
md"""
# SparseArraysBase.jl
- **SparseArraysBase.jl**[^1] elementwise n-dimensional sparse array interface and library, with a built-in dictionary-of-keys array.
  - SparseArrays.jl[^2] Julia standard library is focused on CSC format sparse matrices and vectors.
  - SparseArrayKit.jl[^3] also has an n-dimensional dictionary-of-keys sparse array, but it doesn't allow for generic element types (like a sparse array of arrays, which is helpful for implementing block sparse arrays), and isn't as extensible.
- Designed around an extensible interface based on **DerivableInterfaces.jl**[^4], which allows non-subtypes of `SparseArraysBase.AbstractSparseArray` to opt-in to functionality defined in SparseArraysBase.jl.

[1]: [https://github.com/ITensor/SparseArraysBase.jl](https://github.com/ITensor/SparseArraysBase.jl)

[2]: [https://github.com/JuliaSparse/SparseArrays.jl](https://github.com/JuliaSparse/SparseArrays.jl)

[3]: [https://github.com/Jutho/SparseArrayKit.jl](https://github.com/Jutho/SparseArrayKit.jl)

[4]: [https://github.com/ITensor/DerivableInterfaces.jl](https://github.com/ITensor/DerivableInterfaces.jl)
"""

# ╔═╡ 38673c68-1dda-4658-b5ad-9929102284a5
md"## Construction"

# ╔═╡ 3ab45a2f-b17b-4d0d-97a3-c2a488d82520
a = SparseArrayDOK{Float64}(2, 2)

# ╔═╡ a389e7f9-8137-43c0-b035-1c4cb5517463
a[1, 2] = 12;

# ╔═╡ c09070fb-437e-4394-874d-d1de915a3774
a

# ╔═╡ ee44d85e-33e4-444e-aaed-d4b807e7310d
md"## Properties"

# ╔═╡ c07657d0-fada-4099-b1fa-a6350398caa1
isstored(a, 1, 1)

# ╔═╡ 8ea58194-aa7c-4950-8878-e962c8b047a1
isstored(a, 1, 2)

# ╔═╡ 7e184158-5d27-4020-8772-e6047c2dbac5
eachstoredindex(a) |> collect

# ╔═╡ c1f10b83-4ccd-404f-b42a-098ebcf22a03
md"## Algebra"

# ╔═╡ 1ac8503e-7fa3-401f-81c4-ac0c399b35a9
md"### Addition, permutation, scaling"

# ╔═╡ 446b102b-a66a-48cc-be68-5cb3cf601fa9
2a

# ╔═╡ b10dbf16-047a-437a-b34e-7a8a201f7b82
a'

# ╔═╡ 252988b6-780f-46e7-97c3-40386042a715
a .+ 2 .* a'

# ╔═╡ f54173fa-b013-433e-9f8f-c6e2e9713757
md"### Matrix multiplication"

# ╔═╡ 9f1c4006-943f-4536-bbc4-1639e2317917
a * a'

# ╔═╡ a42299c5-c23c-45f5-93d6-dce3ee5de5e5
md"### Tensor contraction"

# ╔═╡ b3ca394b-6ab0-4677-bb83-80f715a38745
a1 = SparseArrayDOK{Float64}(2, 2, 2);

# ╔═╡ ae8807f7-2324-47f4-9375-50177cf5f2be
a1[1, 2, 1] = 2;

# ╔═╡ 9444aa1f-92e6-428d-86b6-6284a02bf5ea
a1

# ╔═╡ 6ecf0da9-1c48-4d77-bb9e-eae590a2c2d0
a2 = SparseArrayDOK{Float64}(2, 2, 2);

# ╔═╡ b3b0ca1c-dac7-4cbb-8e3b-1f48f36caaf1
a2[2, 1, 2] = 3;

# ╔═╡ 37744e06-edd3-4f95-8cf1-6e2a94915eb8
a2

# ╔═╡ 800a9c3a-8fd5-4a84-b028-09f6342efddf
c, cdimnames = contract(a1, ("i", "j", "k"), a2, ("j", "k", "l"));

# ╔═╡ 651ad156-0f31-467c-9610-7bea7f01869d
c

# ╔═╡ f7c26fda-86e4-4637-a2cb-5636302f24f6
permutedims(a1, (2, 3, 1))

# ╔═╡ 47c85d51-9f2f-4cc1-91fc-cf24b43bad20
cdimnames

# ╔═╡ b292a848-d9c9-45a5-a120-949fbe315eab
md"""
# NamedDimsArrays.jl
- **NamedDimsArrays.jl**[^1] is a generalization of **ITensors.jl**[^2] that provides functionality and interfaces for Julia arrays with names attached to the dimensions.
  - It will serve as the basis for the rewrite of ITensors.jl[^3], where `ITensors.ITensor` will just be a particular type of `NamedDimsArrays.AbstractNamedDimsArray`, and most functionality will be implemented in NamedDimsArrays.jl.
- Note there are a number of Julia packages that provide named dimensions, such as NamedDims.jl[^4] and DimensionalData.jl[^5], but we make different interface and design choices (such as automatically aligning dimensions when the names are not aligned).

[1]: [https://github.com/ITensor/NamedDimsArrays.jl](https://github.com/ITensor/NamedDimsArrays.jl)

[2]: [https://github.com/ITensor/ITensors.jl](https://github.com/ITensor/ITensors.jl)

[3]: [https://github.com/ITensor/ITensors.jl/pull/1611](https://github.com/ITensor/ITensors.jl/pull/1611)

[4]: [https://github.com/invenia/NamedDims.jl](https://github.com/invenia/NamedDims.jl)

[5]: [https://github.com/rafaqz/DimensionalData.jl](https://github.com/rafaqz/DimensionalData.jl)
"""

# ╔═╡ 64aafb37-9384-4b3f-8bca-e62a647d3597
md"## Construction"

# ╔═╡ 684100fa-7646-48e8-9ad6-75e0e090e1bb
a

# ╔═╡ 59d80995-55e3-4672-9cc8-55afe59656cc
aᵢⱼ = NamedDimsArray(a, ("i", "j"))

# ╔═╡ 4b97723c-64c8-4e27-9c3f-18c6a08cc2ef
md"## Operations"

# ╔═╡ a0552b71-91af-41b3-abe8-1a4ff46a94b2
unname(aᵢⱼ)

# ╔═╡ ce739840-568c-4912-89ea-4fb8634cae82
dimnames(aᵢⱼ)

# ╔═╡ 44b63ebd-3919-472e-8931-5faa40dbe515
aᵢⱼ["i" => 1, "j" => 2]

# ╔═╡ c97567ef-1141-4b3e-bea2-9dda16649681
aᵢⱼ["j" => 2, "i" => 1]

# ╔═╡ 3cfd4eda-9c68-405e-ad06-7ba3d339476b
aᵢⱼ["i" => 1:2, "j" => 2]

# ╔═╡ 7b0d5785-25c7-45d6-b403-5d32fba29984
unname(aᵢⱼ["i" => 1:2, "j" => 2])

# ╔═╡ 0f10e645-7b4c-4d63-9e70-44bf8ad2b5fe
@view aᵢⱼ["i" => 1:2, "j" => 2]

# ╔═╡ 70a57992-ee1d-4569-a1c4-28b7aadbb60b
aⱼᵢ = NamedDimsArray(a, ("j", "i"))

# ╔═╡ cb39b4d2-163b-4a2c-9963-a525f14dd3dd
aᵢⱼ + 2aⱼᵢ

# ╔═╡ 1b2fa3e2-c9dd-4311-912c-4d20f31b131b
aᵢⱼ .+ 2 .* aⱼᵢ

# ╔═╡ 67c1d608-6117-45c4-b9a1-7d3fc0608dde
aₖⱼ = NamedDimsArray(a, ("k", "j"))

# ╔═╡ 077804b8-6ec7-4921-93e0-f271aa4a71bd
dimnames(aₖⱼ)

# ╔═╡ 1436b235-4632-4799-9527-071133de55bb
aᵢⱼ * aₖⱼ # Perform tensor contraction, same as: a * a'

# ╔═╡ f7c155ea-3145-4593-a602-df36f64740d4
a * a'

# ╔═╡ 0a8f0c74-d0e5-4c06-bd4c-39587be2a699
dimnames(aᵢⱼ * aₖⱼ)

# ╔═╡ 97dffb52-d83f-4e4c-93a3-1956e67ddcfa
aligndims(aᵢⱼ, ("j", "i")) # Align the dimensions, permute the underlying data

# ╔═╡ 0cd4a492-2f6c-4a3c-8d12-590a6d387f40
aᵢⱼ

# ╔═╡ 5b46d578-92ac-44fd-881d-d1d1eb34b3a7
dimnames(aligndims(aᵢⱼ, ("j", "i")))

# ╔═╡ c9449cf1-7a48-4f68-9d62-aa86e5887840
md"""
# BlockSparseArrays.jl
- **BlockSparseArrays.jl**[^1] based on BlockArrays.jl[^2] and **SparseArraysBase.jl**[^3] (discussed in the previous section):
  - BlockArrays.jl provides a nice block array interface for blockwise indexing/slicing, along with implementations focused on dense block arrays (i.e. where all blocks exist).
  - Designed around a sparse array of arrays, where the sparse arrays are the ones defined in **SparseArraysBase.jl**. Operations make use of the fact that not all blocks exist.
- Symmetric tensors are simply `BlockSparseArrays.BlockSparseArray` objects with symmetry sectors attached to blocked dimensions (i.e. graded vector spaces[^4]), based on **SymmetrySectors.jl**[^5] and **GradedUnitRanges.jl**[^6].

[1]: [https://github.com/ITensor/BlockSparseArrays.jl](https://github.com/ITensor/BlockSparseArrays.jl)

[2]: [https://github.com/JuliaArrays/BlockArrays.jl](https://github.com/JuliaArrays/BlockArrays.jl)

[3]: [https://github.com/ITensor/SparseArraysBase.jl](https://github.com/ITensor/SparseArraysBase.jl)

[4]: [https://en.wikipedia.org/wiki/Graded\_vector\_space](https://en.wikipedia.org/wiki/Graded_vector_space)

[5]: [https://github.com/ITensor/SymmetrySectors.jl](https://github.com/ITensor/SymmetrySectors.jl)

[6]: [https://github.com/ITensor/GradedUnitRanges.jl](https://github.com/ITensor/GradedUnitRanges.jl)
"""

# ╔═╡ 8891acda-396f-42ee-ac73-30b907932905
md"## Construction"

# ╔═╡ 95134d0b-bef2-465d-acef-00220a81adde
b = BlockSparseArray{Float64}([2, 3], [2, 3])

# ╔═╡ 0628d903-91c1-4e19-bb29-6c4465cb53ea
b[2, 4] = 24;

# ╔═╡ 6b98aff2-6139-48c6-9956-a2b5ad5e4ebc
b

# ╔═╡ 2b8a1816-b04b-4cc0-a2b3-1700038cbf9c
b[4, 2] = 42;

# ╔═╡ d1c55959-d134-4085-a3f4-2c9a766b2ebf
b

# ╔═╡ 2ec56d37-fba6-4dc1-8585-f8a309b3ba4e
blocks(b)

# ╔═╡ 6b357db3-fe0c-45a1-afc6-44e35e74b4c0
md"## Indexing/slicing"

# ╔═╡ 61b9fcc9-f65f-4a2c-b0f2-d0bbb2c01c94
b

# ╔═╡ 96014d9c-32ce-464a-9a62-36488c9c18ae
b[Block(1, 1)]

# ╔═╡ 12f3e0a2-991c-45ca-9ea1-b11ceb6a29a6
b[Block(1, 2)]

# ╔═╡ ee2f5908-6d8d-4f39-8a7a-5fdb676f8959
b[Block(1), Block.(1:2)]

# ╔═╡ a8cfdcd4-3458-4e3e-94ef-c425d974d215
md"## Algebra"

# ╔═╡ bc02dd35-0f0a-414c-9f68-c57decef9562
b * b

# ╔═╡ 8d6cf7d4-c396-420e-b2a3-916c55e230b4
b .+ 2 .* b'

# ╔═╡ 6e04e073-3ece-4ced-b36a-4ed76127a567
md"## Named dimensions"

# ╔═╡ f6f158e9-a89d-49f2-aa6f-0c4c5a99fdcb
bᵢⱼ = NamedDimsArray(b, ("i", "j"))

# ╔═╡ f1d915e9-92b3-40d9-bfb4-e8e784fc71da
bⱼᵢ = NamedDimsArray(b, ("j", "i"))

# ╔═╡ ae4e70ad-91d9-4630-a8fe-308713edf36e
bₖⱼ = NamedDimsArray(b, ("k", "j"))

# ╔═╡ 41e74a31-50dd-49cd-9166-fc04cc2b6f99
bᵢⱼ * bₖⱼ

# ╔═╡ a01d4764-7426-45b2-b510-c11f39f20eec
md"""
# Next-generation ITensors.jl
#### Prototyping in ITensorBase.jl[^1]

[1]: [https://github.com/ITensor/ITensorBase.jl](https://github.com/ITensor/ITensorBase.jl)
"""

# ╔═╡ 127545b5-3fa5-4892-b908-aee022dd7411
i, j = Index.((2, 2))

# ╔═╡ fe354580-d530-481c-8ebb-4f63f74a4ab7
x = randn(i, j)

# ╔═╡ 003c379c-0189-4d3f-9578-3afb0c747ebd
typeof(x)

# ╔═╡ f331ef8c-f8d2-492f-9a83-5e32b0569849
x isa AbstractNamedDimsArray

# ╔═╡ 48a93f4f-a854-452d-a180-ee2cd2a198c0
inds(x)

# ╔═╡ 80e9a858-f81d-4230-b2bd-7b5caeaebb16
x[i => 1:2, j => 1]

# ╔═╡ f8acebaf-0a71-4753-b2d0-6d2b3b96f841
inds(x[i => 1:2, j => 1])

# ╔═╡ b751edfa-b1ac-4572-9387-8e48cb1357e9
y = randn(j, i)

# ╔═╡ e832a0ef-debf-4e64-9f9b-9706549010ed
aligndims(y, (i, j))

# ╔═╡ 72f895d3-0c22-4e8d-86df-d65c7d0adbc0
x + y

# ╔═╡ 239e7072-bfeb-4259-a7fa-71bde177ce8c
unname(x) + unname(y)'

# ╔═╡ 490d7025-577c-471f-80bf-513cdb7fd3ce
md"""
# Next steps
- Get to feature completeness with ITensors.jl.
  - Main missing features are block sparse truncated SVD, QR, etc.
  - Abelian symmetric tensors (in progress, a lot is working already, see **SymmetrySectors.jl**[^1] and **GradedUnitRanges.jl**[^2]).
  - Make sure GPU operations work, particularly with block sparse.
  - Test out in ITensorMPS.jl[^3] and ITensorNetworks.jl, make appropriate updates as needed (the biggest update we need to make is upgrading the `OpSum` to `MPO`/`TTN` constructor).
- Continue work on FusionTensors.jl so it acts as a fully working ITensor backend.
- Special thanks to **Karl Pierce**, **Lukas Devos**, **Olivier Gauthé**, and **Miles Stoudenmire** for help on this!

[1]: [https://github.com/ITensor/SymmetrySectors.jl](https://github.com/ITensor/SymmetrySectors.jl)

[2]: [https://github.com/ITensor/GradedUnitRanges.jl](https://github.com/ITensor/GradedUnitRanges.jl)

[3]: [https://github.com/ITensor/ITensorMPS.jl/pull/108](https://github.com/ITensor/ITensorMPS.jl/pull/108)
"""

# ╔═╡ 3d03f743-363f-4efb-a817-6f95e643a6c6
md"""
# Bonus: Abelian symmetries
- In the new design, an Abelian symmetric tensor is simply `BlockSparseArrays.BlockArray` with graded vector spaces as the dimensions.
- Symmetry groups and sectors are defined in **SymmetrySectors.jl**[^1], and graded vector spaces are defined in **GradedUnitRanges.jl**[^2].

[1]: [https://github.com/ITensor/SymmetrySectors.jl](https://github.com/ITensor/SymmetrySectors.jl)

[2]: [https://github.com/ITensor/GradedUnitRanges.jl](https://github.com/ITensor/GradedUnitRanges.jl)
"""

# ╔═╡ f60276b6-562b-4231-b9f9-05272c34b537
r = gradedrange([U1(0) => 2, U1(1) => 2])

# ╔═╡ bf99b143-876a-447d-a915-1f975182cef7
t = BlockSparseArray{Float32}(r, dual(r))

# ╔═╡ 6f2db3b0-c3d8-4e36-bb7e-6e43bf0d7858
t[Block(1, 1)] = randn(2, 2);

# ╔═╡ 8caf1be9-a6ec-47b9-9b0f-522828df1ef4
t[Block(2, 2)] = randn(2, 2);

# ╔═╡ dddc98b7-51d6-4e3f-8c15-37668dcd4381
t

# ╔═╡ bd105b7e-91a9-49e7-910c-a3090c08e0e6
axes(t, 1)

# ╔═╡ e4191b50-92e0-4e0b-8c10-96b4417fd1cc
axes(t, 2)

# ╔═╡ 8068c657-8a36-49ee-a009-bc167979fca7
t * t

# ╔═╡ ba6c2aa7-56e2-4555-8330-8c9a3c70496e
axes(t * t, 1)

# ╔═╡ 2d674f19-51be-420f-b151-624f8a99f48f
axes(t * t, 2)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
BlockSparseArrays = "2c9a651f-6452-4ace-a6ac-809f4280fbb4"
GradedUnitRanges = "e2de450a-8a67-46c7-b59c-01d5a3d041c5"
ITensorBase = "4795dd04-0d67-49bb-8f44-b89c448a1dc7"
Kroki = "b3565e16-c1f2-4fe9-b4ab-221c88942068"
NamedDimsArrays = "60cbd0c0-df58-4cb7-918c-6f5607b73fde"
SparseArraysBase = "0d5efcca-f356-4864-8770-e1ed8d78f208"
SymmetrySectors = "f8a8ad64-adbc-4fce-92f7-ffe2bb36a86e"
TensorAlgebra = "68bd88dc-f39d-4e12-b2ca-f046b68fcc6a"

[compat]
BlockArrays = "~1.3.0"
BlockSparseArrays = "~0.2.2"
GradedUnitRanges = "~0.1.3"
ITensorBase = "~0.1.14"
Kroki = "~1.0.0"
NamedDimsArrays = "~0.4.0"
SparseArraysBase = "~0.2.11"
SymmetrySectors = "~0.1.3"
TensorAlgebra = "~0.1.7"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.3"
manifest_format = "2.0"
project_hash = "24742ea09a0724cd2716b00e6c13093797666590"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "0ba8f4c1f06707985ffb4804fdad1bf97b233897"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.41"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "50c3c56a52972d78e8be9fd135bfb91c9574c140"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.1.1"

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

    [deps.Adapt.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "017fcb757f8e921fb44ee063a7aafe5f89b86dd1"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.18.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra"]
git-tree-sha1 = "2bf6e01f453284cb61c312836b4680331ddfc44b"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "1.11.0"

    [deps.ArrayLayouts.extensions]
    ArrayLayoutsSparseArraysExt = "SparseArrays"

    [deps.ArrayLayouts.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.BlockArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra"]
git-tree-sha1 = "b406207917260364a2e0287b42e4c6772cb9db88"
uuid = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
version = "1.3.0"

    [deps.BlockArrays.extensions]
    BlockArraysBandedMatricesExt = "BandedMatrices"

    [deps.BlockArrays.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"

[[deps.BlockSparseArrays]]
deps = ["Adapt", "ArrayLayouts", "BlockArrays", "DerivableInterfaces", "DiagonalArrays", "Dictionaries", "FillArrays", "GPUArraysCore", "GradedUnitRanges", "LinearAlgebra", "MacroTools", "MapBroadcast", "SparseArraysBase", "SplitApplyCombine", "TypeParameterAccessors"]
git-tree-sha1 = "da110a117d3055cbdb03cdb98697a4cbd8511423"
uuid = "2c9a651f-6452-4ace-a6ac-809f4280fbb4"
version = "0.2.12"
weakdeps = ["LabelledNumbers", "TensorAlgebra"]

    [deps.BlockSparseArrays.extensions]
    BlockSparseArraysGradedUnitRangesExt = "GradedUnitRanges"
    BlockSparseArraysTensorAlgebraExt = ["LabelledNumbers", "TensorAlgebra"]

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "545a177179195e442472a1c4dc86982aa7a1bef0"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.7"

[[deps.CommonWorldInvalidations]]
git-tree-sha1 = "ae52d1c52048455e85a387fbee9be553ec2b68d0"
uuid = "f70d9fcc-98c5-4d4a-abd7-e4cdeebd8ca8"
version = "1.0.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "f36e5e8fdffcb5646ea5da81495a5a7566005127"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.3"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DerivableInterfaces]]
deps = ["Adapt", "ArrayLayouts", "ExproniconLite", "LinearAlgebra", "MLStyle", "MapBroadcast", "TypeParameterAccessors"]
git-tree-sha1 = "1d2da01c828947b2ee5ab730f6c6dd29f6b74136"
uuid = "6c5e35bf-e59e-4898-b73c-732dcc4ba65f"
version = "0.3.14"

[[deps.DiagonalArrays]]
deps = ["ArrayLayouts", "DerivableInterfaces", "FillArrays", "LinearAlgebra", "SparseArraysBase"]
git-tree-sha1 = "f5bac4df77367d0531c34bc18bc081e6f3b9b948"
uuid = "74fd4be6-21e2-4f6f-823a-4360d37c7a77"
version = "0.2.3"

[[deps.Dictionaries]]
deps = ["Indexing", "Random", "Serialization"]
git-tree-sha1 = "1cdab237b6e0d0960d5dcbd2c0ebfa15fa6573d9"
uuid = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
version = "0.4.4"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.EllipsisNotation]]
deps = ["StaticArrayInterface"]
git-tree-sha1 = "3507300d4343e8e4ad080ad24e335274c2e297a9"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.8.0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.ExproniconLite]]
git-tree-sha1 = "c13f0b150373771b0fdc1713c97860f8df12e6c2"
uuid = "55351af7-c7e9-48d6-89ff-24e801d99491"
version = "0.10.14"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GradedUnitRanges]]
deps = ["BlockArrays", "Compat", "FillArrays", "LabelledNumbers", "SplitApplyCombine"]
git-tree-sha1 = "d98b566386fe9330a27e3781b3e3fabb181a4035"
uuid = "e2de450a-8a67-46c7-b59c-01d5a3d041c5"
version = "0.1.3"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "c67b33b085f6e2faf8bf79a61962e7339a81129c"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.15"

[[deps.HalfIntegers]]
git-tree-sha1 = "9c3149243abb5bc0bad0431d6c4fcac0f4443c7c"
uuid = "f0d1745a-41c9-11e9-1dd9-e5d34d218721"
version = "1.6.0"

[[deps.ITensorBase]]
deps = ["Accessors", "DerivableInterfaces", "FillArrays", "LinearAlgebra", "MapBroadcast", "NamedDimsArrays", "UnallocatedArrays", "UnspecifiedTypes", "VectorInterface"]
git-tree-sha1 = "5a6422a3d878c49d270a24b0248c6bb3d5cb9dbc"
uuid = "4795dd04-0d67-49bb-8f44-b89c448a1dc7"
version = "0.1.14"
weakdeps = ["DiagonalArrays", "SparseArraysBase"]

    [deps.ITensorBase.extensions]
    ITensorBaseDiagonalArraysExt = "DiagonalArrays"
    ITensorBaseSparseArraysBaseExt = ["NamedDimsArrays", "SparseArraysBase"]

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.Indexing]]
git-tree-sha1 = "ce1566720fd6b19ff3411404d4b977acd4814f9f"
uuid = "313cdc1a-70c2-5d6a-ae34-0150d3930a38"
version = "1.1.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.Kroki]]
deps = ["Base64", "CodecZlib", "DocStringExtensions", "HTTP", "JSON", "Markdown", "Reexport"]
git-tree-sha1 = "8ff3884b3f5613214b520d6054f8df8ce0de1396"
uuid = "b3565e16-c1f2-4fe9-b4ab-221c88942068"
version = "1.0.0"

[[deps.LabelledNumbers]]
deps = ["Random"]
git-tree-sha1 = "7d618f832ca47c0ae0aefc157ce34ef0a6c78268"
uuid = "f856a3a6-4152-4ec4-b2a7-02c1a55d7993"
version = "0.1.0"
weakdeps = ["BlockArrays"]

    [deps.LabelledNumbers.extensions]
    LabelledNumbersBlockArraysExt = "BlockArrays"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MacroTools]]
git-tree-sha1 = "72aebe0b5051e5143a079a4685a46da330a40472"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.15"

[[deps.MapBroadcast]]
deps = ["BlockArrays", "Compat", "FillArrays"]
git-tree-sha1 = "f735e853fb5dec9e42fc1959deecfe100db4a8f9"
uuid = "ebd9b9da-f48d-417c-9660-449667d60261"
version = "0.1.7"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NamedDimsArrays]]
deps = ["Adapt", "ArrayLayouts", "DerivableInterfaces", "FillArrays", "LinearAlgebra", "MapBroadcast", "Random", "SimpleTraits", "TensorAlgebra", "TypeParameterAccessors"]
git-tree-sha1 = "b310178db8b4679f887ea47621ab9d6f22a0e990"
uuid = "60cbd0c0-df58-4cb7-918c-6f5607b73fde"
version = "0.4.0"
weakdeps = ["BlockArrays"]

    [deps.NamedDimsArrays.extensions]
    NamedDimsArraysBlockArraysExt = "BlockArrays"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7493f61f55a6cce7325f197443aa80d32554ba10"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.15+3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SparseArraysBase]]
deps = ["Accessors", "ArrayLayouts", "DerivableInterfaces", "Dictionaries", "FillArrays", "LinearAlgebra", "MapBroadcast"]
git-tree-sha1 = "dffcccc4f0be87825884812094a5a90b06ca4fbe"
uuid = "0d5efcca-f356-4864-8770-e1ed8d78f208"
version = "0.2.11"

[[deps.SplitApplyCombine]]
deps = ["Dictionaries", "Indexing"]
git-tree-sha1 = "c06d695d51cfb2187e6848e98d6252df9101c588"
uuid = "03a91e81-4c3e-53e1-a0a4-9c0c8f19dd66"
version = "1.2.3"

[[deps.Static]]
deps = ["CommonWorldInvalidations", "IfElse", "PrecompileTools"]
git-tree-sha1 = "87d51a3ee9a4b0d2fe054bdd3fc2436258db2603"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "1.1.1"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Static"]
git-tree-sha1 = "96381d50f1ce85f2663584c8e886a6ca97e60554"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.8.0"

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

    [deps.StaticArrayInterface.weakdeps]
    OffsetArrays = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.SymmetrySectors]]
deps = ["BlockArrays", "GradedUnitRanges", "HalfIntegers", "LabelledNumbers"]
git-tree-sha1 = "0dd1adee246747bec9f89f6b7c2b40a3340c7214"
uuid = "f8a8ad64-adbc-4fce-92f7-ffe2bb36a86e"
version = "0.1.3"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TensorAlgebra]]
deps = ["ArrayLayouts", "BlockArrays", "EllipsisNotation", "LinearAlgebra", "TupleTools", "TypeParameterAccessors"]
git-tree-sha1 = "8e2d6d893b476ea3fc50324dba18f9dd5831f883"
uuid = "68bd88dc-f39d-4e12-b2ca-f046b68fcc6a"
version = "0.1.7"
weakdeps = ["GradedUnitRanges"]

    [deps.TensorAlgebra.extensions]
    TensorAlgebraGradedUnitRangesExt = "GradedUnitRanges"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.TupleTools]]
git-tree-sha1 = "41e43b9dc950775eac654b9f845c839cd2f1821e"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.6.0"

[[deps.TypeParameterAccessors]]
deps = ["LinearAlgebra", "SimpleTraits"]
git-tree-sha1 = "46866cb947ce400253fda43f5f545d0744a9e37e"
uuid = "7e5a90cf-f82e-492e-a09b-e3e26432c138"
version = "0.2.2"

    [deps.TypeParameterAccessors.extensions]
    TypeParameterAccessorsFillArraysExt = "FillArrays"
    TypeParameterAccessorsStridedViewsExt = "StridedViews"

    [deps.TypeParameterAccessors.weakdeps]
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    StridedViews = "4db3bf67-4bd7-4b4e-b153-31dc3fb37143"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnallocatedArrays]]
deps = ["Adapt", "FillArrays", "TypeParameterAccessors", "UnspecifiedTypes"]
git-tree-sha1 = "94b65c1a93ed6e575c8f5a57b942f1c71eaa43eb"
uuid = "43c9e47c-e622-40fb-bf18-a09fc8c466b6"
version = "0.1.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnspecifiedTypes]]
git-tree-sha1 = "6abc7c7a09e74ae7d46efd94654e4cfddacb8020"
uuid = "42b3faec-625b-4613-8ddc-352bf9672b8d"
version = "0.1.3"

[[deps.VectorInterface]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9166406dedd38c111a6574e9814be83d267f8aec"
uuid = "409d34a3-91d5-4945-b6ec-7529ddf182d8"
version = "0.5.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"
"""

# ╔═╡ Cell order:
# ╟─6cbb26d9-10c7-4f3b-bfc6-03dc5fb53a34
# ╟─957bab60-86a5-4cbb-84f8-e51ec0ce8723
# ╟─75070daa-0985-4886-95a7-b671f6e9005f
# ╟─e973f9cb-c8ca-4e3c-afc2-adce2fe9f1f9
# ╟─f1385767-d637-42f1-827e-9aadd1ddfed7
# ╟─79fa6a5e-ffaa-46b9-97a3-390d045e543f
# ╟─ad59d756-1554-4931-a0fe-dbc4d41d9d42
# ╟─c604e497-24b4-499d-a66e-0e04e3f4cded
# ╟─34c5e555-3919-42a1-af28-103edfeb9140
# ╟─bc6f11e4-2fde-4a2d-9d53-2bd7d5c497fe
# ╟─38673c68-1dda-4658-b5ad-9929102284a5
# ╠═19579878-a2fc-46b6-9a20-b3af72ae2464
# ╠═3ab45a2f-b17b-4d0d-97a3-c2a488d82520
# ╠═a389e7f9-8137-43c0-b035-1c4cb5517463
# ╠═c09070fb-437e-4394-874d-d1de915a3774
# ╟─ee44d85e-33e4-444e-aaed-d4b807e7310d
# ╠═327ef53d-3fd9-4740-b670-b7b29a6a5adb
# ╠═c07657d0-fada-4099-b1fa-a6350398caa1
# ╠═8ea58194-aa7c-4950-8878-e962c8b047a1
# ╠═7e184158-5d27-4020-8772-e6047c2dbac5
# ╟─c1f10b83-4ccd-404f-b42a-098ebcf22a03
# ╟─1ac8503e-7fa3-401f-81c4-ac0c399b35a9
# ╠═446b102b-a66a-48cc-be68-5cb3cf601fa9
# ╠═b10dbf16-047a-437a-b34e-7a8a201f7b82
# ╠═252988b6-780f-46e7-97c3-40386042a715
# ╟─f54173fa-b013-433e-9f8f-c6e2e9713757
# ╠═9f1c4006-943f-4536-bbc4-1639e2317917
# ╟─a42299c5-c23c-45f5-93d6-dce3ee5de5e5
# ╠═19b1594c-4260-4c7c-91f1-eae0013b51e0
# ╠═b3ca394b-6ab0-4677-bb83-80f715a38745
# ╠═ae8807f7-2324-47f4-9375-50177cf5f2be
# ╠═9444aa1f-92e6-428d-86b6-6284a02bf5ea
# ╠═6ecf0da9-1c48-4d77-bb9e-eae590a2c2d0
# ╠═b3b0ca1c-dac7-4cbb-8e3b-1f48f36caaf1
# ╠═37744e06-edd3-4f95-8cf1-6e2a94915eb8
# ╠═800a9c3a-8fd5-4a84-b028-09f6342efddf
# ╠═651ad156-0f31-467c-9610-7bea7f01869d
# ╠═f7c26fda-86e4-4637-a2cb-5636302f24f6
# ╠═47c85d51-9f2f-4cc1-91fc-cf24b43bad20
# ╟─b292a848-d9c9-45a5-a120-949fbe315eab
# ╟─64aafb37-9384-4b3f-8bca-e62a647d3597
# ╠═a042962b-0a5b-426d-83d1-68daa7518ff1
# ╠═684100fa-7646-48e8-9ad6-75e0e090e1bb
# ╠═59d80995-55e3-4672-9cc8-55afe59656cc
# ╟─4b97723c-64c8-4e27-9c3f-18c6a08cc2ef
# ╠═ca1cc56f-3400-43eb-811f-40bcd49a8438
# ╠═a0552b71-91af-41b3-abe8-1a4ff46a94b2
# ╠═ce739840-568c-4912-89ea-4fb8634cae82
# ╠═44b63ebd-3919-472e-8931-5faa40dbe515
# ╠═c97567ef-1141-4b3e-bea2-9dda16649681
# ╠═3cfd4eda-9c68-405e-ad06-7ba3d339476b
# ╠═7b0d5785-25c7-45d6-b403-5d32fba29984
# ╠═0f10e645-7b4c-4d63-9e70-44bf8ad2b5fe
# ╠═70a57992-ee1d-4569-a1c4-28b7aadbb60b
# ╠═cb39b4d2-163b-4a2c-9963-a525f14dd3dd
# ╠═1b2fa3e2-c9dd-4311-912c-4d20f31b131b
# ╠═67c1d608-6117-45c4-b9a1-7d3fc0608dde
# ╠═077804b8-6ec7-4921-93e0-f271aa4a71bd
# ╠═1436b235-4632-4799-9527-071133de55bb
# ╠═f7c155ea-3145-4593-a602-df36f64740d4
# ╠═0a8f0c74-d0e5-4c06-bd4c-39587be2a699
# ╠═97dffb52-d83f-4e4c-93a3-1956e67ddcfa
# ╠═0cd4a492-2f6c-4a3c-8d12-590a6d387f40
# ╠═5b46d578-92ac-44fd-881d-d1d1eb34b3a7
# ╟─c9449cf1-7a48-4f68-9d62-aa86e5887840
# ╟─8891acda-396f-42ee-ac73-30b907932905
# ╠═5e2dd3f1-de9f-416d-b259-6658f2781483
# ╠═95134d0b-bef2-465d-acef-00220a81adde
# ╠═0628d903-91c1-4e19-bb29-6c4465cb53ea
# ╠═6b98aff2-6139-48c6-9956-a2b5ad5e4ebc
# ╠═2b8a1816-b04b-4cc0-a2b3-1700038cbf9c
# ╠═d1c55959-d134-4085-a3f4-2c9a766b2ebf
# ╠═7edf8314-0645-4e52-afc6-f9ac3fe8de9a
# ╠═2ec56d37-fba6-4dc1-8585-f8a309b3ba4e
# ╟─6b357db3-fe0c-45a1-afc6-44e35e74b4c0
# ╠═8cffacf7-b87c-4a8a-9c2f-edeeb791bbd5
# ╠═61b9fcc9-f65f-4a2c-b0f2-d0bbb2c01c94
# ╠═96014d9c-32ce-464a-9a62-36488c9c18ae
# ╠═12f3e0a2-991c-45ca-9ea1-b11ceb6a29a6
# ╠═ee2f5908-6d8d-4f39-8a7a-5fdb676f8959
# ╟─a8cfdcd4-3458-4e3e-94ef-c425d974d215
# ╠═bc02dd35-0f0a-414c-9f68-c57decef9562
# ╠═8d6cf7d4-c396-420e-b2a3-916c55e230b4
# ╟─6e04e073-3ece-4ced-b36a-4ed76127a567
# ╠═f6f158e9-a89d-49f2-aa6f-0c4c5a99fdcb
# ╠═f1d915e9-92b3-40d9-bfb4-e8e784fc71da
# ╠═ae4e70ad-91d9-4630-a8fe-308713edf36e
# ╠═41e74a31-50dd-49cd-9166-fc04cc2b6f99
# ╟─a01d4764-7426-45b2-b510-c11f39f20eec
# ╠═1e35e904-fc55-45df-b7cd-4996744fce16
# ╠═127545b5-3fa5-4892-b908-aee022dd7411
# ╠═fe354580-d530-481c-8ebb-4f63f74a4ab7
# ╠═003c379c-0189-4d3f-9578-3afb0c747ebd
# ╠═c836c0d2-1685-4a83-a436-d251c31d8c15
# ╠═f331ef8c-f8d2-492f-9a83-5e32b0569849
# ╠═48a93f4f-a854-452d-a180-ee2cd2a198c0
# ╠═80e9a858-f81d-4230-b2bd-7b5caeaebb16
# ╠═f8acebaf-0a71-4753-b2d0-6d2b3b96f841
# ╠═b751edfa-b1ac-4572-9387-8e48cb1357e9
# ╠═e832a0ef-debf-4e64-9f9b-9706549010ed
# ╠═72f895d3-0c22-4e8d-86df-d65c7d0adbc0
# ╠═239e7072-bfeb-4259-a7fa-71bde177ce8c
# ╟─490d7025-577c-471f-80bf-513cdb7fd3ce
# ╟─3d03f743-363f-4efb-a817-6f95e643a6c6
# ╠═fb236256-490a-4524-a5f3-36ecdf927fe0
# ╠═3a90675f-1a5a-4fc2-8698-96c40394467b
# ╠═f60276b6-562b-4231-b9f9-05272c34b537
# ╠═bf99b143-876a-447d-a915-1f975182cef7
# ╠═6f2db3b0-c3d8-4e36-bb7e-6e43bf0d7858
# ╠═8caf1be9-a6ec-47b9-9b0f-522828df1ef4
# ╠═dddc98b7-51d6-4e3f-8c15-37668dcd4381
# ╠═bd105b7e-91a9-49e7-910c-a3090c08e0e6
# ╠═e4191b50-92e0-4e0b-8c10-96b4417fd1cc
# ╠═8068c657-8a36-49ee-a009-bc167979fca7
# ╠═ba6c2aa7-56e2-4555-8330-8c9a3c70496e
# ╠═2d674f19-51be-420f-b151-624f8a99f48f
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
