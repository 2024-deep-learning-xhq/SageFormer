# export CUDA_VISIBLE_DEVICES=0

model_name=SageFormer
seq_len=192
cls_len=2
graph_depth=4
knn=16
e_layers=3


python -u run.py \
  --task_name long_term_forecast_new \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path day.csv \
  --model_id day_192_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 96 \
  --pred_len 96 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 23 \
  --dec_in 23 \
  --c_out 23 \
  --n_heads 4 \
  --cls_len $cls_len \
  --graph_depth $graph_depth \
  --knn $knn \
  --batch_size 16 \
  --learning_rate 1e-5 \
  --lradj 'type3' \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 20 \
  --patience 5