DATASET=NaturalQuestions # [NaturalQuestions, PopQA, TriviaQA, ASQA]

CUDA_VISIBLE_DEVICES=0 python src/inference.py \
  --dataset_name $DATASET \
  --do_qa_statement_generation \
  --n_docs 0 \
  --output_dir dataset/${DATASET} \
  --model_dir meta-llama/Meta-Llama-3-8B-Instruct \
  --prompt_dict_path src/qa.json \
  --max_new_tokens 80