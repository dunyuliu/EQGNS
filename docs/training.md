# Training EQGNS (MeshNet)

Three entry points, lowest- to highest-level:

## 1. Direct

```shell
python3 -m meshnet.train \
  --data_path=<working_dir>/dataset/ \
  --model_path=<working_dir>/models.<suffix>/ \
  --batch_size=2 --ntraining_steps=3000000 --nsave_steps=100000
```

Resume by adding `--model_file=latest --train_state_file=latest`.

## 2. Single-run wrapper: `train_cli.py`

```shell
python3 train_cli.py <dataset-name> <model_suffix> <gpu_id> <batch_size> \
  [--ntraining_steps N] [--nsave_steps N] [--model_file latest --train_state_file latest]
# example
python3 train_cli.py case3.200m nmp10.cotopaxi 0 2
```

Assumes the working directory `./gns-sample/<dataset-name>/` and sets
`CUDA_VISIBLE_DEVICES` and `OMP_NUM_THREADS=1` for you.

## 3. Hyperparameter sweep: `run.process.gns.py`

```shell
python3 run.process.gns.py \
  --working_dir <working_dir> \
  --mode train --model_suffix r1 --gpu_id 0 \
  --ntraining_steps 1000000 --nsave_steps 100000 \
  --learning_rates 1e-4,3e-5 \
  --batch_sizes 2 --noise_stds 0.005,0.02 \
  --nmessage_passing_steps 5,10 --machine_name knox
```

Runs the Cartesian product of the listed hyperparameters. Each combination
gets a model directory named
`models.<suffix>_lr<lr>_bs<bs>_ns<noise>_nmp<nmp>_<machine>/` and training
auto-resumes from `latest` if checkpoints already exist there.

## config.json

`meshnet/train.py` loads `<model_path>/config.json` at startup (a template is
in `meshnet/example.config.json`). Key fields:

- `INPUT_SEQUENCE_LENGTH`, `dt`, `noise_std`
- `lr_init`, `lr_decay_rate`, `lr_decay_steps`
- `simulator_nmessage_passing_steps`, `simulator_latent_dim`,
  `simulator_nmlp_layers`, `simulator_mlp_hidden_dim`
- `simulator_nnode_types`, `simulator_node_type_embedding_size`,
  `simulator_nnode_in`, `simulator_nedge_in`

The sweep driver rewrites this file per run, so the config stored next to a
checkpoint always records the settings that produced it.

For the hyperparameters used in the published results, see the paper
(doi:10.1029/2025JB031981) and the archived configs on Zenodo
(doi:10.5281/zenodo.17095311). Loss curves can be inspected with
`utils/plot.loss.curve.py` or `utils/plot.loss.curve.complex.py`.
