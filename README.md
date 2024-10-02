# PEAR: POSITION-EMBEDDING-AGNOSTIC ATTENTION RE-WEIGHTING ENHANCES RETRIEVAL-AUGMENTED GENERATION WITH ZERO INFERENCE OVERHEAD

The codes are implemented using PyTorch. The code for detecting heads, which suppresses the models' context awareness, is based on the work of Wang et al. and Lv et al. I have fixed some bugs, made it more efficient for inference, and enabled model-parallelism to support multiple GPUs. We have completed the assembly of Baichuan and set up a proxy task focused on context copying.


## Detect heads

```bash
python path_llama.py
```
## Coefficients Re-weight

### train
```bash
sh train.sh
```
## evaluation
The experimental validation code consists of two parts: the multi-document multi-position MDQA task and the long-context QA task, referring to the experimental setup of "lost-in-the-middle".

### MDQA
test:
```bash
sh mdqa_llama.sh
```
eval:
```bash
python eval.py
```
### long-context QA
test:
```bash
CUDA_VISIBLE_DEVICES=0 python all_longma_pred.py --model longma2-30
```

eval:
```bash
python em_eval.py --model model
```
