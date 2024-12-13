# export CUDA_VISIBLE_DEVICES=0

model_name=Informer
seq_len=192
cls_len=1
graph_depth=3
knn=16
e_layers=2


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path 5min.csv \
  --model_id 5min_192_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 96 \
  --pred_len 96 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 22 \
  --dec_in 22 \
  --c_out 22 \
  --n_heads 4 \
  --cls_len $cls_len \
  --graph_depth $graph_depth \
  --knn $knn \
  --batch_size 16 \
  --learning_rate 2e-5 \
  --lradj 'type3' \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 10 \
  --patience 4