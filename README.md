# PEAR: POSITION-EMBEDDING-AGNOSTIC ATTENTION RE-WEIGHTING ENHANCES RETRIEVAL-AUGMENTED GENERATION WITH ZERO INFERENCE OVERHEAD

The codes are implemented using PyTorch. The code for detecting heads, which suppresses the models' context awareness, is based on the work of [Wang et al.](https://github.com/redwoodresearch/Easy-Transformer) and [Lv et al.](https://github.com/trestad/Factual-Recall-Mechanism) We have completed the assembly of Baichuan-13B-chat and set up a proxy task focused on context copying.


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
The experimental validation code consists of two parts: the multi-document multi-position MDQA task and the long-context QA task, referring to the experimental setup of [lost-in-the-middle]([https://github.com/redwoodresearch/Easy-Transformer](https://github.com/nelson-liu/lost-in-the-middle)).

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
