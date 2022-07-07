## Fine-tuning Wav2Vec2

```bash
python3.8 run_common_voice.py\ 
--model_name_or_path="facebook/wav2vec2-large-xlsr-53"\ 
--dataset_config_name="fa"\ 
--output_dir=./wav2vec2-large-xlsr-fa-demo\ 
--overwrite_output_dir\ 
--num_train_epochs="5"\ 
--per_device_train_batch_size="1"\ 
--learning_rate="3e-4"\ 
--warmup_steps="500"\ 
--evaluation_strategy="steps"\ 
--save_steps="400"\ 
--eval_steps="400"\ 
--logging_steps="400"\ 
--save_total_limit="3"\ 
--freeze_feature_extractor\ 
--feat_proj_dropout="0.0"\ 
--layerdrop="0.1"\ 
--gradient_checkpointing\ 
--fp16\ 
--group_by_length\ 
--do_train\ 
--do_eval\ 
--per_device_train_batch_size=4 
```
Provided by Moein Salimi
<br>
(Reference: Wav2Vec2 Finetuning Code)
