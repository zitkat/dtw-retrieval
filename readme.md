# Dynamic time warping for retrieval of variable length sequences

## Plan
1. DTW and adaptive pooling experiments:
   1. Image retrieval on IAM, features from:
   - our custom transformer
   - Seq2seq
   - ResNet
   2. Speech retrieval?
   3. NLP retrieval?
2. Metric learning dtw
3. Quantization for variable length sequences

# TODO:
- generate resnet50 activations
- parallelize dtwn
- use soft dtw
- analysis
 
## How to run retrieval experiment

File `retrieval_settings.csv` contains individual named configurations
for the experiments, pass the names of these to the `exact_retrieval_experiment.py`

For IAM dataset run as:
```shell
> python exact_retrieval_experiment.py -i RWTH.iam_word_gt_final.test.retrieval_1.csv  -c "../ocr/checkpoints/transformer_words.model" -s retrieval_settings.csv -sv enc_mf_apo3_1 
```

- `-c` option can be used to override checkpoint from settings
- checkpoint, layer and model name from settings are used to infer name of the file with saved embeddings
- "RWTH.iam_word_gt_final.test.retrieval_1.csv" must contain list of samples with filenames, labels and frequencies