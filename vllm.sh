CUDA_VISIBLE_DEVICES=2,3 vllm serve /home/jxliu/ext0/models/Qwen3-8B \
    --host 0.0.0.0 \
    --port 8011 \
    --served-model-name /home/jxliu/ext0/models/Qwen3-8B \
    --trust-remote-code \
    --tensor-parallel-size 2 \