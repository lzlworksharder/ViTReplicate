export HF_ENDPOINT=https://hf-mirror.com
pip install -r requirements.txt
huggingface-cli download uoft-cs/cifar100 --repo-type dataset --local-dir ./data/cifar100
huggingface-cli download google/vit-large-patch16-224 --local-dir ./pretrained/vit-l-16