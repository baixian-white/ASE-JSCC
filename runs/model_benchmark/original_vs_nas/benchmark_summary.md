# Model Benchmark Summary

- model_a: `original` (legacy)
- model_b: `nas_retrain` (nas)

| scope | A mean±std | B mean±std | delta(B-A) | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| AWGN | 0.9190±0.0000 | 0.9762±0.0000 | 0.0571 | ±0.0000 |
| Fading | 0.9190±0.0000 | 0.9833±0.0025 | 0.0643 | ±0.0016 |
| Combined_channel | 0.9181±0.0030 | 0.9857±0.0000 | 0.0676 | ±0.0019 |
| overall_mean | 0.9187±0.0010 | 0.9817±0.0008 | 0.0630 | ±0.0009 |
