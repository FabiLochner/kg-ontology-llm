"""
fact_extraction_text.py

This script extracts basic facts from text with an LLM.

"""

# Load libraries
from datetime import date, datetime
from dotenv import load_dotenv
import json 
import os 
from pathlib import Path
from openai import OpenAI
import argparse


# Load env variables
load_dotenv()


## 0) Parse command-line arguments

parser = argparse.ArgumentParser(description="Extract facts from text file with LLM (structured outputs).")
parser.add_argument("--model", type=str,  required=True,  help="LLM model name (e.g. gpt-5.6-luna)")
parser.add_argument("--prompt_template", type=str,  required=True,  help="Path to prompt template text file (e.g. scripts/evaluation/text_gold_standard/prompts/default_kg-gen_mine_1.txt)")
parser.add_argument("--run", type=str,  required=True,  help="Run identifier for repeated runs of the same LLM (e.g. run_1, run_2, run_3)")
parser.add_argument("--text", type = str, required = True, help = "Path to text file facts will be extracted from (e.g., data/raw/transcripts/lfDJDNRh5Iw_de.txt)")
parser.add_argument("--out_dir", type=str, default="data/interim/text/facts", help="Directory for the output JSONL file")
parser.add_argument("--reasoning_effort", type=str, default="medium", help="Reasoning effort for reasoning models")

args = parser.parse_args()

model = args.model 
prompt_template = args.prompt_template
run = args.run
text = args.text


## 1) Load LLM extraction prompt 

def load_prompt_template(path: str) -> str:
    """Loads an LLM prompt to extract facts from text."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

llm_prompt = load_prompt_template(prompt_template)
source_text = Path(args.text).read_text(encoding="utf-8")
source_id = Path(args.text).stem


# Load and verify prompt template and source text
print(f"\n--- Loaded prompt template: {args.prompt_template} ---")
print(f"--- Source: {source_id} ({len(source_text.split())} words) ---")



## 2) LLM annotation (currently: OpenAI API with structured outputs)


## Set API & LLM configs 
openai_api_key = os.getenv("openai_api_key")
openai_api_base = os.getenv("openai_api_url")

sample_non_reasoning_models = {"gpt-4.1-mini", "gpt-4o-mini"} #adding open-source non reasoning models later

## Function to set LLM configs based on reasoning vs non-reasoning nature of LLM
## if non reasoning -> aim for reproducability (seed, top p, temperature)
## if reasoning -> temperature = 1 by default and not changeable

def build_params(model, *, temperature = 0, seed = 42, reasoning_effort = "medium"): #set medium as default for reasoning models
    p = {"model": model}
    if any (model.startswith(m) for m in sample_non_reasoning_models):
        p |= {"temperature": temperature, "top_p": 1.0, "seed": seed}
    else:
        p |= {"reasoning_effort": reasoning_effort}
    return p


## Structured Output Schema

FACTS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "extracted_facts",
        "strict": True,
        "schema": {
            "type": "object",
            "properties":{
                "facts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Short sentences, each describing how one object relates to another."
                }
            },
            "required": ["facts"],
            "additionalProperties": False,  
        },
    },
}



## Start OpenAI client
client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

## Function to extract facts 

response = client.chat.completions.create(
    messages = [{"role": "user", "content": f"{llm_prompt}\n\n{source_text}"}], # default LLM MINE-1 prompt does not have a system prompt, only user prompt
    response_format=FACTS_SCHEMA,
    **build_params(args.model, reasoning_effort=args.reasoning_effort),
)

message = response.choices[0].message

# The schema guarantees valid JSON, but the model can still refuse, will leave the
# content unusable, so check before trusting it.
if getattr(message, "refusal", None):
    raise RuntimeError(f"Model refused: {message.refusal}")

 
facts = json.loads(message.content)["facts"]
