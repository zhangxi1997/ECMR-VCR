# models
## Replicating validation results
Here's how you can replicate my val results. Run the command(s) below. First, you might want to make your GPUs available. When I ran these experiments I used

`source activate CMR && export CUDA_VISIBLE_DEVICES=0`

- For question answering, run:
```
python my_train.py -params CMR_sGCN_attr/default.json -folder saves/flagship_answer -plot saves/flagship_answer
```

- for Answer justification, run
```
python my_train.py -params CMR_sGCN_attr/default.json -folder saves/flagship_rationale -plot saves/flagship_rationale -rationale
```

You can combine the validation predictions using
`python eval_q2ar.py -answer_preds saves/flagship_answer/valpreds.npy -rationale_preds saves/flagship_rationale/valpreds.npy`

