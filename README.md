# AIGER Model Checker

A Rust-based model checker for formal verification of digital circuits specified in the AIGER ASCII format (`.aag`).
This tool provides both bounded model checking (BMC) and interpolation-based unbounded model checking to verify **safety properties**.

The program accepts AIGER circuits with a _single_ output. As per the conventions of the [AIGER](https://fmv.jku.at/aiger/) format, 
this output gate encodes the safety property to be verified. Thus, a counterexample to the property is a series of input signals
across consecutive time steps such that the output gate becomes `1` (true).

## Features

The tool uses a slightly modified version of the open-source [MiniSat SAT solver](http://minisat.se) (v1.14p) to apply bounded model checking
techniques. We use version 1.14p because this is the only MiniSat release that supports proof logging, which we utilize to
compute interpolants as over-approximations of the reachable state set.

In particular, the application offers two modes:

1. **Bounded Model Checking (BMC)**:  
Checks if a property violation can occur within a given fixed number of time steps $k$.
Therefore, an output of `OK` indicates that the property holds within the given bound (making no claim about any bound $k' > k$).

2. **Interpolation-Based Model Checking**:  
Performs incremental model checking with interpolation-based fixpoint detection
to provide unbounded safety proofs (or find counterexamples). The algorithm again starts from a fixed unwinding bound $k$
and performs bounded model checking. However, if this shows that the property is safe within the current bound, the 
algorithm computes an [interpolant](https://en.wikipedia.org/wiki/Craig_interpolation) from the resulting resolution proof
to over-approximate the set of states reachable in $k+1$ steps. This over-approximation is then used to check again for
a reachable violation of the property.  
We continue this loop until we either find a genuine counterexample (property violation even after the over-approximation is discarded)
or reach a fixpoint (the new interpolant logically implies the previous one). The latter implies that we have considered all possible states
the circuit can take within _any_ number of steps, thus proving the **property is safe for a _arbitrary_ bound**.

---

## Execution

### Prerequisites

The project requires:
- Rust (2024 edition or later)
- A C++ compiler (`gcc`, `clang` or `msvc`) and the standard library of `C++11`

### Building the Project

Clone the repository and build the project using Cargo:

```powershell
cargo build --release
```

This builds an executable called `bmc` (bounded model checking) located at `target/release/bmc`.

**Note:** The C++ code does _not_ need to be compiled manually. Compilation and linking of the project is all done via the Cargo build
system. Please only ensure that a compatible C++ compiler can be located on your system by `cc` ([see docs](https://docs.rs/cc/latest/cc/)).

I suggest to build the project on WSL if you are on Windows.

---

## Usage

Run the model checker by providing a bound $k$ and the path to an AIGER file.

```powershell
./bmc [OPTIONS] <k> <AAG_FILE>
```

### Arguments

- `<k>`: Number of unwinding steps to the transition relation.
- `<AAG_FILE>`: Path to an AIGER file in ASCII format (`*.aag`).

**Note:** The tool only supports the ASCII format of AIGER (`*.aag`). To use the more widely used binary format (`*.aig`),
please convert it to ASCII using the [aigtoaig](https://fmv.jku.at/aiger/) tool first.

### Options

- `-i`, `--interpolate`: Perform bounded model checking incrementally with interpolation-based fixpoint detection.
This mode automatically increases the unwinding depth $k$ until a fixpoint or a counterexample is found.
- `-v`, `--verbose`: Enables additional print statements for progress information and prints a **witness** if the
property could be violated.
- `-h`, `--help`: Print help information.

### Examples

**Standard Bounded Model Checking:**
Check if the property is violated within 10 steps:
```sh
./target/release/bmc 10 data/count10.aag
```

**Interpolation-Based Unbounded Checking:**
Prove safety or find a counterexample starting with an initial bound of `k = 3`:
```sh
./target/release/bmc -i 3 data/safe.aag
```

### Outputs
- `OK`: The property holds within the given bound (BMC mode) or is proven safe for _any_ bound (interpolation mode).
- `FAIL`: A counterexample to the safety property was found.

Use the `-v` flag to print a counterexample if the property is violated.

---

## Project Structure

```text
.
├── Cargo.toml             # Rust project configuration
├── src/                   # Source code
│   ├── main.rs              # CLI entry point and argument parsing
│   ├── bmc/                 # Core model checking logic
│   │   ├── aiger.rs           # AIGER ASCII (*.aag) parser
│   │   ├── model.rs           # BMC and interpolation-based checking
│   │   └── debug.rs           # Debug utilities and witness printing
│   └── logic/               # SAT solving and interpolation logic
│       ├── solving.rs         # SAT solver abstraction
│       ├── resolution.rs      # Resolution proof logging and interpolation
│       ├── types.rs           # Data structures for propositional logic
│       └── minisat/           # C++ MiniSat source files + custom bindings
└── data/                  # Example AIGER circuits (*.aag) for testing
```

## Author

Johannes Riedmann
