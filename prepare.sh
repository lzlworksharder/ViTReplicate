export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download uoft-cs/cifar100 --repo-type dataset --local-dir ./data/cifar100
huggingface-cli download google/vit-large-patch16-224 --local-dir ./pretrained/vit-l-16