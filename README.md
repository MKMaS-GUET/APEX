# APEX: daptive Variable-wise Parallel Execution for Worst-Case Optimal Joins on Graph Queries

APEX is a graph query engine for RDF data based on worst-case optimal joins (WCOJ).
It supports variable-wise parallel execution and dynamic variable ordering, and is designed to efficiently process complex graph queries on large-scale knowledge graphs.

---

## Features

* Build RDF databases with trie-based indexes
* Execute graph queries using worst-case optimal joins
* Train and test variable ordering models
* Multi-threaded query execution

---

## Build

1. Clone this project

```shell
git clone git@github.com:MKMaS-GUET/RDF-TDAA.git
git submodule update --init
```

2. Build this project

```shell
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Or use the `build.sh` script to build this project directly

```shell
./scripts/build.sh
```

---

## Usage

### Build Database

```bash
./apex build -d <DATABASE_PATH> -f <RDF_DATA_FILE>
```

### Query Database

```bash
./apex query -d <DATABASE_PATH> -f <QUERY_FILE> -t <NUM_THREADS>
```

### Train Model

```bash
./apex train -d <DATABASE_NAME> -f <QUERY_FILE> -t <NUM_THREADS>
```

### Test Model

```bash
./apex test -d <DATABASE_NAME> -f <QUERY_FILE> -t <NUM_THREADS>
```

---

## Options

| Option             | Description                 |
| ------------------ | --------------------------- |
| `-d`, `--database` | Database path or name       |
| `-f`, `--file`     | RDF data file or query file |
| `-t`, `--threads`  | Number of threads           |

---

## Example

```bash
./apex build -d mydb -f data.nt
./apex query -d mydb -f queries.txt -t 16
```

---

## Notes

* If the database path does not contain `/`, it will be stored under `DB_DATA_ARCHIVE/` by default.
* The default number of threads is 32.