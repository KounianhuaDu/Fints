# Data Preparation
See under [data_process/].
The preparation process is for [Headline Generation](https://lamp-benchmark.github.io/), [Abstract Writing](https://longlamp-benchmark.github.io/), and [PersonalWAB](https://github.com/HongruCai/PersonalWAB)

Run
```
./new_data_process.py
```
to select some users for experiments.
Run
```
./ranking.py --task [your_task]
```
to sort all historical samples according to their relevance to the corresponding training or testing samples.

# Negative Samples Generation
For PersonalWAB, run
```
python data_collect_pwab.py --dataset pwab \
--modelweight [root_of_models] \
--k 5 \
--llm llama-3.1 \
--form json \
--data_path ../pa_back/data
```
to generate positive samples.

For all datasets, run
```
./data_collect.sh
```
to generate negtive samples.

# Vectors Generation and Evaluation
Run
```
./run.sh
```
to generate personalized steering vectors and evaluate the model on the test set.

After evaluation, for PersonalWAB, the evaluation scores will be saved; for other datasets, add the parameter `--eval `in `run_generation.py` and run again to obtain the model's evaluation scores.

By adding `--plugin` in `run_generation.py`, PA-steering can be used with lora models.


