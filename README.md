# PEAR: POSITION-EMBEDDING-AGNOSTIC ATTENTION RE-WEIGHTING ENHANCES RETRIEVAL-AUGMENTED GENERATION WITH ZERO INFERENCE OVERHEAD

The codes are implemented using PyTorch. The code for detecting heads, which suppresses the models' context awareness, is based on the work of Wang et al. and Lv et al. I have fixed some bugs, made it more efficient for inference, and enabled model-parallelism to support multiple GPUs. We have completed the assembly of Baichuan and set up a proxy task focused on context copying.


## Detect heads

```bash
python path_llama.py
bash
```
## Coefficients Re-weighting

train

bash
```bash
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
