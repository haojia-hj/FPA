DATASET=NaturalQuestions # [NaturalQuestions, PopQA, TriviaQA, ASQA]

CUDA_VISIBLE_DEVICES=0 python src/inference.py \
  --dataset_name $DATASET \
  --n_docs 5 \
  --output_dir dataset/${DATASET} \
  --model_dir your_fine-tuned_model \
  --prompt_dict_path src/rag.json \
  --max_new_tokens 4096