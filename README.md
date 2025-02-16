# Bad Improved Prompt Model

This repository presents a fine-tuned version of the Llama 3.2 3B model, specifically trained to enhance the quality of user prompts. By leveraging a dataset of poorly constructed prompts paired with their improved counterparts, the model aims to assist in refining vague or unclear prompts provided by users.

## Model

The fine-tuned model is available on Hugging Face:

- **Model Name**: [bad-improved-prompt-model](https://huggingface.co/milliyin/bad-improved-prompt-model)

This model is developed by milliyin and is licensed under the Apache-2.0 license. It is fine-tuned from the model `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`.

## Dataset

The model was trained on the following dataset:

- **Dataset Name**: [bad-improved-prompt-pair-sharegpt](https://huggingface.co/datasets/milliyin/bad-improved-prompt-pair-sharegpt)

This dataset contains pairs of bad and improved prompts, facilitating the model's learning process to enhance prompt quality. It is available in Parquet format and consists of approximately 11,500 rows.

## Fine-Tuning

The fine-tuning process is documented in the following Jupyter Notebook:

- **Fine-Tuning Notebook**: [finetuning.ipynb](https://github.com/milliyin/bad-improved-prompt-model/blob/main/finetuning.ipynb)

This notebook provides a step-by-step guide on how the model was fine-tuned using the dataset mentioned above.

## Testing the Model

To evaluate the performance of the fine-tuned model, refer to the following Jupyter Notebook:

- **Model Testing Notebook**: [finetune_model_test.ipynb](https://github.com/milliyin/bad-improved-prompt-model/blob/main/finetune_model_test.ipynb)

This notebook demonstrates how to test the model's ability to improve user prompts.

## Usage

To utilize the fine-tuned model, you can use the following code:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "milliyin/bad-improved-prompt-model"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_8bit=True)

from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
)
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

messages = [
    {"role": "system", "content": "You are the assistant who converts bad sentences into grammatically correct, well-structured, detailed, and lengthy sentences."},
    {"role": "user", "content": "Model wears red dress"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,  # Must add for generation
    return_tensors="pt",
).to("cuda")

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=128,
    use_cache=True,
    temperature=1.5,
    min_p=0.1
)

print(tokenizer.batch_decode(outputs))
```

Replace `"Model wears red dress"` with the prompt you wish to improve. The model will generate an enhanced version of the input prompt.

**Note**: Ensure that the necessary libraries (`transformers`, `torch`, and `unsloth`) are installed in your environment. Additionally, verify that your system has a compatible GPU for model inference.

## License

The model and dataset are licensed under the Apache-2.0 license.

Made with ❤️ by milliyin
