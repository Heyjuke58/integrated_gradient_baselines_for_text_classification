# integrated_gradients_baselines_for_text_classifications
Examination of the use of different baseline techniques for integrated gradients using BERT as a case study


## How to run

- install conda environment as provided by `environment.yml`
- Precompute KNN:

```sh
python knn.py
```

- Find out about the CLI options:

```sh
python main.py --help
```

## Badges

- Implementation: Models are initialized in `main.py`, baselines are defined in `src/baseline_builder.py`. Baselines can be chosen with the `-b=BASELINE(S)` option.
- Visualization (Sum of Cumulative gradients): Done in `visualization.py: visualize_attrs()`. On by default.
- Visualization (Path): TODO
- Analysis: `src/baselines` contains baseline definitions for 11 different baselines.
- Evaluation: Comprehensiveness and Log-odds are implemented in `src/ablation_evaluation.py`. Visualization on by default. It is recommended to use many samples for this, e.g: `python main.py --e=0-50`
- Extension: Implementation from codebase of original paper in `src/dig.py`. DIG only works with discrete baselines (`pad_embed`, `furthest_word`, `avg_word`). DIG can be toggled on by using the `-v=dig` option. The DIG strategy can be chosen with the `--dig-strategy=STRATEGY` option.