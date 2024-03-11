import torch
from peft import  PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,  GenerationConfig
import warnings

warnings.filterwarnings("ignore")

trained_model_dir = 'pet.pth'
config = PeftConfig.from_pretrained(trained_model_dir)

model_name = "PY007/TinyLlama-1.1B-step-50K-105b"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

config = PeftConfig.from_pretrained(trained_model_dir)

trained_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    # quantization_config=bnb_config,
    trust_remote_code=True,
    device_map='auto'

    )


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='right')
tokenizer.pad_token = tokenizer.eos_token

def get_test_prompt(query,context,role):
    # Определяем список колонок, которые будем использовать


    # Формируем текст промптинга
    prompt = f"[INST]\n"
    prompt += "Use the given context to guide your answer about the query as indicated by your role:\n\n"
    prompt += f"Query: {query}\n\n"
    prompt += f"Context: {context}\n\n"
    prompt += f"Your role: {role}\n\n"
    prompt += "Answer:[/INST]"

    return  prompt


device = 'cuda' if torch.cuda.is_available() else 'cpu'
def generate_answer(query, model):
    encoding = tokenizer(query, return_tensors="pt").to(device)
    generation_config = GenerationConfig(max_new_tokens=250, pad_token_id = tokenizer.eos_token_id,repetition_penalty=1.3, eos_token_id = tokenizer.eos_token_id)
    outputs = model.generate(input_ids=encoding.input_ids, generation_config=generation_config)
    text_output = tokenizer.decode(outputs[0],skip_special_tokens=True)
    return text_output.split("[/INST]")[1].split("\n\n")[1]


def answer(question, context):
    role='Homer Simpson'
    prompt = get_test_prompt(question,context,role)
    answer = generate_answer(prompt, trained_model)

    return answer
