# integrated_gradients_baselines_for_text_classifications
Examination of the use of different baseline techniques for integrated gradients using BERT as a case study


## How to run

- install conda environment as provided by `environment.yml` and activate it.

```sh
conda env create --file environment.yml
conda activate iml-project
```

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
- Visualization (Path): Closest-by embeddings are calculated with `token_embedding_helper.py: get_closest_by_token_embed_for_embed()`,  Visualization (PCA and as a table) in `visualization.py: visualize_word_paths() and visualize_word_path_table()`. Off by default. Option: --viz-word-path
- Analysis: `src/baselines` contains baseline definitions for 11 different baselines.
- Evaluation: Comprehensiveness and Log-odds are implemented in `src/ablation_evaluation.py`. Visualization on by default. It is recommended to use many samples (or at least 50) for this, e.g: `python main.py --e=0-50`
- Extension: Implementation from codebase of original paper in `src/dig.py`. DIG only works with discrete baselines (`pad_embed`, `furthest_word`, `avg_word`). DIG can be toggled on by using the `-v=dig` option. The DIG strategy can be chosen with the `--dig-strategy=STRATEGY` option.

## Further Requirements

- Unit tests in folder `test`. We tested our baseline generation, our own helper functions and the ablation evaluation
- Random seed runs for uniform baseline can be found in folder `figures_randomness`