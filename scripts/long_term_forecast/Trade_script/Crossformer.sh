seq_len=192
model_name=Crossformer

root_path_name=./dataset/weather/
data_path_name=5min.csv
model_id_name=5min
data_name=custom

random_seed=2021
for pred_len in 96
do
    python -u run_former.py \
      --random_seed $random_seed \
      --seasonal_patterns Monthly \
      --task_name 'long_term_forecast'\
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 22 \
      --dec_in 22 \
      --c_out 22 \
      --e_layers 3 \
      --n_heads 16 \
      --d_ff 256 \
      --dropout 0.2\
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --lradj 'type3'\
      --pct_start 0.2\
      --itr 1 --batch_size 16 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done