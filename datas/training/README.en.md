# Training workloads

[Simplified Chinese](README.md)

The default workload for training tests and benchmarks is the 192K/CP16
manifest under `real/`: each global pack contains 196,608 tokens, the CP size
is 16, and each rank receives twelve 1,024-token chunks.

`generate_cases.py` generates sanitized files from a local source
CSV. The 1,500 call records normalize to 1,253 unique shapes, while duplicate
selection references and weights preserve the real frequency. Generated files
contain only `case_id`, `cu_seqlens`, shape metrics, and test/benchmark
selections. They do not retain dataset or source-sample identifiers. The
original CSV is excluded by `.gitignore`.

```bash
python3 datas/training/generate_cases.py
python3 datas/training/generate_cases.py --check
```

Legacy synthetic 128K/CP16, 256K/CP32, and 512K/CP64 workloads remain under
`synthetic/`. They provide optional supplementary coverage and are excluded from the default
test and benchmark selections.
