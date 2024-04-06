from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import torch
from dotenv import load_dotenv
import os
import re # type: ignore
from tavily import TavilyClient
from threading import Thread # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, TextIteratorStreamer
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
tavily_api_key = os.environ["TAVILY_API_KEY"]
tavily = TavilyClient(api_key=tavily_api_key)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = "left"

    streamer = TextIteratorStreamer(tokenizer, skip_prompt = True)

    compute_dtype = getattr(torch, "bfloat16")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
        cache_dir=f"./models",
        local_files_only=True
    )
    
    return model, tokenizer, streamer

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
model, tokenizer, streamer = get_model(model_name)

def get_generation_kwargs(instruction, stream = False):
    chat = [s.strip() for s in re.split("\[INST\]|\[\/INST\]", instruction)[1:-1]]
    context = tavily.get_search_context(query=chat[-1], search_depth="advanced", max_tokens=1500)
    chat[-1] = f"{chat[-1]}\nThe following context which are search results from internet which may be related to the given query. Use the context to answer the current question if you need.\nContext: {context}"

    instruction = "[INST] " + chat[0] + " [/INST]"
    for i in range(1, len(chat), 2):
        instruction += " " + chat[i] + " [INST] " + chat[i+1] + " [/INST] "

    prompt = instruction.strip()
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()

    if len(input_ids[0]) >= model.config.max_position_embeddings:
        raise Exception("My context limit reached, start a new chat session by refreshing the page.")
    
    if stream:
        return dict(
            input_ids=input_ids,
            generation_config=GenerationConfig(pad_token_id=tokenizer.pad_token_id, temperature=1.0, top_p=1.0, top_k=50, num_beams=1),
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=256,
            streamer=streamer
        )
    
    return dict(
            input_ids=input_ids,
            generation_config=GenerationConfig(pad_token_id=tokenizer.pad_token_id, temperature=1.0, top_p=1.0, top_k=50, num_beams=1),
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=256
        )

def model_generate(instruction):
    if instruction == "":
        return "Invalid input"
    
    try:
        generation_kwargs = get_generation_kwargs(instruction, stream=True)
    except Exception as e:
        return str(e)

    generation_output = model.generate(**generation_kwargs)

    response = ""

    for seq in generation_output.sequences:
        output = tokenizer.decode(seq)
        response += output.strip()

    return response

def model_generate_stream(instruction):
    if instruction == "":
        yield "Invalid input"
        return

    try:
        generation_kwargs = get_generation_kwargs(instruction, stream=True)
    except Exception as e:
        yield str(e)
        return

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text

@app.get("/")
async def root():
    return "Hello world!"

@app.get("/generate")
async def generate(prompt: str = ""):
    return model_generate(prompt)

@app.get("/generate-stream")
async def generate_stream(prompt: str = ""):
    return StreamingResponse(model_generate_stream(prompt))