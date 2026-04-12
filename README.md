# Implementation
This is the repo for 
<img width="659" height="260" alt="image" src="https://github.com/user-attachments/assets/3828ccd0-e96f-42b2-ab9e-5b95968b160c" />


## Data Preparation
See under [data_process/].
The preparation process is for [Headline Generation](https://lamp-benchmark.github.io/), [Abstract Writing](https://longlamp-benchmark.github.io/), and [PersonalWAB](https://github.com/HongruCai/PersonalWAB)

Run
```
./new_data_process.py
```
to select some users for experiments. Modify the data file path and number of users if needed.

Run
```
./ranking.py --task [your_task]
```
to sort all historical samples according to their relevance to the corresponding training or testing samples.

## Negative Samples Generation
For PersonalWAB, run
```
python data_collect_pwab.py --dataset pwab \
--modelweight [root_of_models] \
--k 5 \
--llm llama-3.1 \
--form json \
--data_path [path to your data]
```
to generate positive samples.

For all datasets, run
```
./data_collect_para.sh
```
to generate negtive samples.

## Vectors Generation and Evaluation
Run
```
./run.sh
```
to generate personalized steering vectors and evaluate the model on the test set. 

After evaluation, for PersonalWAB, the evaluation scores will be saved; for other datasets, add the parameter `--eval `in `run_generation.py` and run again to obtain the model's evaluation scores.

## Other baselines
- OPPU: Run `OPPU-main/run_OPPU.sh`
- PerPcs: Run `Per-Pcs-main/run_PerPcs.sh`
- ReLLa: Run `ReLLa/run.sh`
- SynthesizeMe: Run 
```
bash ./vllm.sh
python ./SynthesizeMe.py \
    --dataset LaMP_4 \
    --port 8010 \
    --best-of-n 2 \
```





