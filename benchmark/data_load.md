Below is a suggested **README.md** file, formatted and organized for clarity. Adjust any details as needed for your specific environment or preferences.

---

# TigerGraph Data Load

This guide outlines how to load **OGBN Products** and **OGBN Papers100M** datasets into TigerGraph (TG). The steps include creating the necessary schema in TigerGraph, preparing your data files, and running the load process.

---

## OGBN Products

### Step 1: Create a Graph in TigerGraph

Create a graph in TigerGraph with the following schema structure:

- **Node Type**: `product`  
  Attributes:
  ```plaintext
  "product": {
      "embedding": "LIST<Float>",  # to store the node features
      "node_label": INT,
      "train_val_test": INT  # stores the split enum (0: train, 1: valid, 2: test)
  }
  ```
- **Edge Type**: `rel`  
  A directed edge from `product` to `product`:
  ```plaintext
  "rel": "product" -> "product"
  ```

### Step 2: Prepare the Data

1. **Download the product data** (e.g., from the OGB repository).
2. Ensure each CSV file has a **node ID**. For example:
   - `raw/node-feat.csv`: Add a node ID column (0 to n for each row) alongside the feature columns.
   - `raw/node-label.csv`: Add a node ID column (0 to n) alongside the label column.
   - `split/train.csv`: Add a `split` column with value **0** for all rows (indicating training nodes).
   - `split/valid.csv`: Add a `split` column with value **1** for all rows (indicating validation nodes).
   - `split/test.csv`: Add a `split` column with value **2** for all rows (indicating test nodes).
   - `raw/edges.csv`: No change needed (already contains edge pairs).

### Step 3: Load the Data into TigerGraph

Use the sample script (e.g., `benchmark/ogbn_dataload.py`) or your own loading process to ingest CSV files into TigerGraph. Ensure your file paths and column separators match your CSV format. 

Below is an **example** load job snippet that inserts node features (assuming CSV format: `node_id | features`):

```tql
LOAD f1 TO VERTEX product VALUES($0, SPLIT($2, ","), _, _) 
USING SEPARATOR="|", HEADER="true", EOL="\n", QUOTE="DOUBLE";
```

Adjust the column references (`$0`, `$2`, etc.) as needed to match your CSV structure.

---

## OGBN Papers100M

### Step 1: Create a Graph in TigerGraph

Create a graph in TigerGraph with the following schema structure:

- **Node Type**: `paper`  
  Attributes:
  ```plaintext
  "paper": {
      "embedding": "LIST<Double>",  # to store the node features
      "node_label": INT,
      "train_val_test": INT  # stores the split enum (0: train, 1: valid, 2: test)
  }
  ```
- **Edge Type**: `cites`  
  A directed edge from `paper` to `paper`:
  ```plaintext
  "cites": "paper" -> "paper"
  ```

### Step 2: Prepare the Data

1. **Download the papers data** (e.g., from the OGB repository).
2. Convert `.npz` files to CSV:
   - `raw/data.npz` → parse into `node-feat.csv` and `edges.csv`  
     - **node-feat.csv**: add a node ID column (0 to n).  
     - **edges.csv**: no change required (already contains edge pairs).
   - `raw/node-label.npz` → parse into `node-label.csv` (also add node ID).
3. Add split columns for train/valid/test:
   - `split/time/train.csv`: add a `split` column with value **0**.  
   - `split/time/valid.csv`: add a `split` column with value **1**.  
   - `split/time/test.csv`: add a `split` column with value **2**.

### Step 3: Load the Data into TigerGraph

Use the sample script (e.g., `ogbn_load.py`) or your own process to load each CSV file into TigerGraph. Make sure you provide correct file paths and column definitions.

Below is an **example** load job snippet that inserts node features:

```tql
LOAD f1 TO VERTEX paper VALUES($0, SPLIT($2, ","), _, _) 
USING SEPARATOR="|", HEADER="true", EOL="\n", QUOTE="DOUBLE";
```

(Adjust the column references (`$0`, `$2`, etc.) as needed to match your CSV structure.

---

**Note**: Always validate that your schema, CSV format, and load script align properly to avoid data ingestion errors. For more details on TigerGraph loading jobs, refer to official TigerGraph documentation or your local environment’s sample scripts.