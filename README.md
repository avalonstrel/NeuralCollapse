# NeuralCollapse: Reproduction for [A Law of Data Separation in Deep Learning](https://arxiv.org/abs/2210.17020)

Reproduction for A Law of Data Separation in Deep Learning, mainly focus on MLP cases.

### Main Results

![Main Results](https://github.com/avalonstrel/NeuralCollapse/blob/mlp/figures/mlp/results/optims_layers_stn.png)
Frist two layers corresponds to SGD and Momentum optimization, the third row means the network with Layer normalization.
Columns represents the numbers of the MLP.
We can find the results fit the conclusion in the paper.
### Running
#### Training Models from Scratch

```python
python3 train.py --config ./exprs/expr_prefix/config_name/config.yaml \
                                --work-dir workdir \
                                --data-dir datadir \
                                --gpu-ids gpu_id \
                                --seed seed &

```

You can refer to the bash script **train.sh** about how to use the **train.py**, you only need to change the value of **datadir**.

#### Computing the Measures for Features from Each Layers

```python
python3 analysis_revised.py --model_names config_name \
                                        --gpu_id gpu_id \
                                        --seeds seed \
                                        --feat_types tag \
                                        --expr_prefix expr_prefix \
                                        --metric_types metric_types \
                                        --data-dir datadir &

```

You can refer to the bash script **analysis.sh** about how to use the **analysis.py**, you only need to change the values of **datadir** ,**config_names**.
Here **config_names** means the config names for the models you want to analyze.
