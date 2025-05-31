DATASET=NaturalQuestions # [NaturalQuestions, PopQA, TriviaQA, ASQA]

CUDA_VISIBLE_DEVICES=0 python src/form_pre_data.py \
  --dataset_name $DATASET \
  --n_docs 5 \
  --output_dir dataset/${DATASET} \
  --model_dir meta-llama/Meta-Llama-3-8B-Instruct \
  --max_new_tokens 4096 \
  --max_instances 7000