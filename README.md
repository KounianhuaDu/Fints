# Data Preparation
See under [data_process/].
The preparation process is for [LaMP_4](https://lamp-benchmark.github.io/), abstract generation and [PersonalWAB](https://github.com/HongruCai/PersonalWAB)

Run
```
./new_data_process.py
```
to select some users for experiments.
Run
```
./ranking.py --task LaMP_4
```
to sort all historical samples according to their relevance to the corresponding training or testing samples.

# Negative Samples Generation
For PersonalWAB, run
```
python data_collect_pwab.py --dataset pwab \
--modelweight /inspire/hdd/global_user/zhangweinan-24046 \
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


