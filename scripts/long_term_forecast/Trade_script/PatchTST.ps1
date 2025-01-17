$seq_len = 192
$model_name = "PatchTST"
$root_path_name = ".\dataset\weather\"
$data_path_name = "5min.csv"
$model_id_name = "5min"
$data_name = "custom"
$random_seed = 2021

foreach ($pred_len in 96) {
  python -u run_TST.py `
    --random_seed $random_seed `
    --seasonal_patterns Monthly `
    --task_name 'long_term_forecast' `
    --is_training 1 `
    --root_path $root_path_name `
    --data_path $data_path_name `
    --model_id ${model_id_name}_${seq_len}_${pred_len} `
    --model $model_name `
    --data $data_name `
    --features M `
    --seq_len $seq_len `
    --pred_len $pred_len `
    --enc_in 22 `
    --e_layers 3 `
    --n_heads 16 `
    --d_model 128 `
    --d_ff 256 `
    --dropout 0.2 `
    --fc_dropout 0.2 `
    --head_dropout 0 `
    --patch_len 16 `
    --stride 8 `
    --des 'Exp' `
    --train_epochs 100 `
    --patience 10 `
    --lradj 'type3' `
    --pct_start 0.2 `
    --itr 1 --batch_size 16 --learning_rate 0.0001 > logs/LongForecasting/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log
}