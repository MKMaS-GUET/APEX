# APEX: Adaptive Variable-wise Parallel Execution for Worst-Case Optimal Joins on Graph Queries

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
git clone git@github.com:MKMaS-GUET/APEX.git
```

2. Install dependencies

```
git submodule update --init
sudo apt install libtbb-dev
```

3. Build this project

```shell
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Binary output: `./bin/apex`

Or use the `build.sh` script to build this project directly

```shell
./scripts/build.sh
```

---

## Usage

```bash
./bin/apex <command> [options]
```

> `apex_server` is no longer built as a standalone executable.  
> Use `apex server` instead.

### Build Database

```bash
./bin/apex build -d <DATABASE_PATH> -f <RDF_DATA_FILE>
```

### Query Database

```bash
./bin/apex query -d <DATABASE_PATH> -f <QUERY_FILE> -t <NUM_THREADS>
```

### Start HTTP Server

```bash
./bin/apex server --host <HOST> --port <PORT> --threads <NUM_THREADS> --max-upload-mb <MAX_MB>
```

### Train Model

```bash
./bin/apex train -d <DATABASE_NAME> -f <QUERY_FILE> -t <NUM_THREADS>
```

### Test Model

```bash
./bin/apex test -d <DATABASE_NAME> -f <QUERY_FILE> -t <NUM_THREADS>
```

---

## Options

| Option             | Command                  | Description                      |
| ------------------ | ------------------------ | -------------------------------- |
| `-d`, `--database` | `build/query/train/test` | Database path or name            |
| `-f`, `--file`     | `build/query/train/test` | RDF data file or query file      |
| `-t`, `--threads`  | `query/train/test`       | Number of query execution threads |
| `--host`           | `server`                 | Server bind host (default `0.0.0.0`) |
| `--port`           | `server`                 | Server bind port (default `8080`) |
| `--threads`        | `server`                 | Query threads used by HTTP API (default `32`) |
| `--max-upload-mb`  | `server`                 | Max upload payload in MB (default `20480`) |

---

## Example

```bash
./bin/apex build -d mydb -f data.nt
./bin/apex query -d mydb -f queries.txt -t 16
./bin/apex server --host 127.0.0.1 --port 8080
```

---

## Notes

* If the database path does not contain `/`, it will be stored under `DB_DATA_ARCHIVE/` by default.
* The default number of threads is 32.
* The server UI is available at `http://<host>:<port>/apex/ui/` after `apex server` starts.
