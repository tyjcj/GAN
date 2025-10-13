```

python AIGgan.py \
  --dataset-dir data_files/datasets/ISCAS85/graph \
  --gen-ckpt   results/ISCAS85/train_results/generator_epoch9.pt \
  --out        results/ISCAS85/AIGfake/sample_0.pt \
  --candidate-k 64
```


```
python utilities/pt2graphml.py -i results/ISCAS85/AIGfake/sample_0.pt \
-o results/ISCAS85/graphml_fake/sample_0.graphml


python utilities/graphml2bench.py --graphml results/ISCAS85/graphml_fake/sample_0.graphml --out results/ISCAS85/bench_fake/sample_0.bench
```

```
python utilities/scripts_check.py --pt_file results/ISCAS85/AIGfake/sample_0.pt --graphml_file results/ISCAS85/graphml_fake/sample_0.graphml --bench_file results/ISCAS85/bench_fake/sample_0.bench
abc -c "read_bench results/ISCAS85/bench_fake/sample_0.bench; strash; print_stats"
```