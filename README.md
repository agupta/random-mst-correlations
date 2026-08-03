# Pairwise edge correlations in random minimum spanning trees

This repository is the GitHub home for the paper
*Pairwise edge correlations in random minimum spanning trees: a universal bound
and complete-graph negative correlation* by Anish Gupta.

The paper proves a universal factor-8 bound for two-edge inclusion in minimum
spanning trees with i.i.d. atomless edge weights, gives simple-graph
counterexamples to pairwise negative correlation, proves strict pairwise
negative correlation on every complete graph, and relates the complete-graph
pair probabilities to the expected minimum-spanning-tree weight.

- **[Read the paper](paper/main.pdf)**
- [Citable preprint v1 on Zenodo](https://doi.org/10.5281/zenodo.21780630)
- [See the preferred citation](CITATION.cff)
- [Explore the exact supporting data](data/kn-exact-table.csv)

Status: preprint, August 2026. This repository provides a stable paper landing
page and its optional supporting materials; it is not a claim of peer review.

## Supporting computations

The mathematical proofs are contained in the paper and do not require the
software. The repository also includes exact data and standard-library Python
checks for readers who want to inspect the finite computations and
certificates.

Requirements are Python 3 using only the standard library and a working
`pdflatex`. No BibTeX or `latexmk` step is required.

~~~sh
make check          # standard exact suite
make check-full     # widest documented exact suite
make table          # regenerate data/kn-exact-table.csv
make clean paper    # clean three-pass manuscript build
make verify-sha256  # verify the release-tree integrity manifest
~~~

The executable checks independently evaluate finite MST measures, regenerate
the complete-graph data, verify polynomial certificates and counterexamples,
and retain deliberately false variants as hostile controls.

## Repository contents

~~~text
paper/                         manuscript source and PDF
data/                          exact and published-comparison tables
src/                           exact algorithms and table generator
tests/                         exact regression and claim checks
evidence/SHA256SUMS.txt        release-tree integrity manifest
CITATION.cff                   GitHub and general citation metadata
LICENSES.md                    manuscript/data/software licence boundary
~~~

## Citation

The citable preprint v1 is archived at
[doi:10.5281/zenodo.21780630](https://doi.org/10.5281/zenodo.21780630).
Preferred citation metadata are provided in `CITATION.cff`.

## Licence

Software under `src/` and `tests/`, together with the `Makefile`, is available
under the MIT License. The manuscript and accompanying data are available under
CC BY 4.0. See [LICENSES.md](LICENSES.md) for the exact boundary.

## Contact

- Anish Gupta, independent researcher
- `ag2269@cantab.ac.uk`
- [ORCID 0009-0008-8137-7729](https://orcid.org/0009-0008-8137-7729)
