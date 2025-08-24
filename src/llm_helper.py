from unsloth import FastModel

max_seq_length = 1024
system_prompt = """You are an expert specialized in medicine. \
Your role is to provide precise, factual, and immediate \
information without any superfluous language, greetings, or acknowledgments.\
"""

def invoke_llm(user_prompt, model, tokenizer):
    "a wrapper function to tokenize/template query and invoke llm "

    messages = [
        {'role': 'system','content': system_prompt},
        {"role" : 'user', 'content' : user_prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True,
    ).removeprefix('<bos>')

    output_tokens = model.generate(
        **tokenizer(text, return_tensors = "pt").to("cuda"),
        max_new_tokens = 500,
        temperature = 1, top_p = 0.95, top_k = 5,
    )
    generated_text = tokenizer.decode(output_tokens[0])
    return generated_text.split('<start_of_turn>model\n')[-1].split('<end_of_turn>')[0]
    
