Below is a formatted `README.md` example. Feel free to adjust or customize to your needs!

---

# Benchmark

## Standalone

### On CPU Workload

Run the `gcn_cpu.py` 
example:
```bash
python gcn_cpu.py
```

### On GPU Workload

Run `gcn_dist_mnmg.py` using `torchrun`:
example:
```bash
torchrun --nnodes 1 --nproc-per-node 4 --rdzv-id 4RANDOM \
         --rdzv-backend c10d --rdzv-endpoint localhost:29500 gcn_dist_mnmg.py
```

## TG Integration

Run the `tg_gcn.py` using `torchrun`:
example:
```bash
torchrun --nnodes 1 --nproc-per-node 4 --rdzv-id 4RANDOM \
         --rdzv-backend c10d --rdzv-endpoint localhost:29500 examples/tg_gcn.py \
         -g ogbn_products \
         -p "your_password" \
         --host "tg_host_ip"
```

- The `-g` argument specifies the graph on which you want to run the benchmark.  
- **Note**: We expect you have already created this graph in the TigerGraph database with the appropriate schema.
- For more information on how to load data into TigerGraph, check `Tg_data_load.md`.




