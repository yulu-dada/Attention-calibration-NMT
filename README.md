# Attention-calibration-NMT

Code for ACL2021 paper "Attention Calibration for Transformer in Neural Machine Translation".

### Requirements
```
Fairseq-0.9
Pytorch-1.6
Python-3.6
```
### Train
We implement the described models with [fairseq toolkit](https://github.com/pytorch/fairseq) for training and evaluating. 
The bilingual datasets should be first preprocessed into binary format and save in 'data-bin-chen' file. More details can refer to the tutorial of Fairseq.

'mycode-fix', 'mycode-anneal' and 'mycode-gate' are corresponding to our proposed three calibration methods 'Fixed Weighed Sum', 'Annealing Learning' and 'Gating Mechanism' as mentioned in section 3.2.
```
user_dir=./mycode-gate/
data_bin=./data-bin-chen/
model_dir=./models/chen-gate-0.03
export CUDA_VISIBLE_DEVICES=0,1
nohup fairseq-train $data_bin \
        --user-dir $user_dir --criterion my_label_smoothed_cross_entropy --task attack_translation_task --arch my_arch \
        --optimizer myadam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt \
        --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.0005 --min-lr 1e-09 \
        --weight-decay 0.0 --label-smoothing 0.1 \
        --max-tokens 8192 --no-progress-bar --max-update 150000 \
        --log-interval 100 --save-interval-updates 1000 --keep-interval-updates 10 --save-interval 10000 --seed 1111 \
        --ddp-backend no_c10d \
        --dropout 0.3 \
        -s ch -t en --save-dir $model_dir \
        --mask-loss-weight 0.03 > log.train-chen-9 &
```
### Inference
```
fairseq-generate $test_path 
        --user-dir ./mycode \
        --criterion my_label_smoothed_cross_entropy --task attack_translation_task --path $model_path \
        --remove-bpe -s $src -t $tgt --beam 4 --lenpen 0.6 \
```
