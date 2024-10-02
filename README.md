# PEAR: POSITION-EMBEDDING-AGNOSTIC ATTENTION RE-WEIGHTING ENHANCES RETRIEVAL-AUGMENTED GENERATION WITH ZERO INFERENCE OVERHEAD

The codes are implemented based on PyTorch. The codes for detecting heads which suppress the models’ context awareness, is based on Wang et al{} and Lv{}. I fixed some bugs, made it more efficient for inference, and enabled model-parallel to support multi-GPUs. 我们完成了对 Baichuan 的组装，并设置了一个代理任务 (a proxy task) focused on context copying。

## Detect heads

代码：
```bash
python path_llama.py
train
代码：

bash
复制代码
sh train.sh
evaluation
实验验证的代码由两部分组成，分别是多文档多位置的 MDQA 任务和长上下文 QA 任务，我们参考了 lost-in-the-middle{} 的实验设置。

MDQA
test:
代码：

bash
复制代码
sh mdqa_llama.sh
eval:
代码：

bash
复制代码
python eval.py
长上下文QA
test:
代码：

bash
复制代码
CUDA_VISIBLE_DEVICES=0 python all_longma_pred.py --model longma2-30
eval:
代码：

bash
复制代码
python em_eval.py --model model
