"""Re-send 6 smoke-test prompts per model and record exact token usage."""
import glob, json, random, sys
from dotenv import load_dotenv
from openai import OpenAI
from src.llm_gate.inference.system_prompts import get_system_prompt

load_dotenv()
client = OpenAI()
random.seed(0)
mds = sorted(glob.glob('benchmark/*/*/step_*/gate_full/cells.md'))
picks = random.sample(mds, 6)
cfg = {'gpt-5.4': dict(temperature=0.0), 'gpt-6-luna': dict(temperature=1.0)}
rows = []
for model, kw in cfg.items():
    for p in picks:
        r = client.chat.completions.create(
            model=model, max_completion_tokens=32768,
            messages=[{'role': 'system', 'content': get_system_prompt()},
                      {'role': 'user', 'content': open(p).read()}], **kw)
        u = r.usage
        rows.append(dict(model=model, md=p, prompt=u.prompt_tokens,
                         cached=u.prompt_tokens_details.cached_tokens,
                         completion=u.completion_tokens,
                         reasoning=u.completion_tokens_details.reasoning_tokens))
        print(json.dumps(rows[-1]), flush=True)
json.dump(rows, open('jobs/logs/usage_sample.json', 'w'), indent=1)
