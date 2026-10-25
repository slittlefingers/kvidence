"""Resident MLX intent extractor. Only JSON is written to stdout; no command execution."""
import json
from pathlib import Path
import sys
import time

SYSTEM = """Extract facts from a user message for a local reconstruction case.
Return one JSON object: {"intent":"create|update|run|status|pause|resume|methods|help","case_name":null,"fields":{}}.
Allowed fields: model_name (gemma4-e2b, gemma3-1b, unknown), observation (path), methods (list of guided, collision_plus, exhaustive, collision_plus_cpa), calibration (path), notes (text), max_positions (integer), max_queries (integer).
Ridge/岭回归=guided. 穷举=exhaustive. Collision+=collision_plus. Official calibrated Collision+=collision_plus_cpa.
Do not guess missing facts. Answers to questions are updates. A method selection or path is NOT a request to run.
Only an explicit start request has intent run. Output JSON only, no explanation."""
EXAMPLES = [
    ('创建案件case2','{"intent":"create","case_name":"case2","fields":{}}'),
    ('我不知道模型是什么','{"intent":"update","case_name":null,"fields":{"model_name":"unknown"}}'),
    ('Gemma 4 E2B','{"intent":"update","case_name":null,"fields":{"model_name":"gemma4-e2b"}}'),
    ('岭回归和穷举','{"intent":"update","case_name":null,"fields":{"methods":["guided","exhaustive"]}}'),
    ('观测目录是 /tmp/evidence','{"intent":"update","case_name":null,"fields":{"observation":"/tmp/evidence"}}'),
    ('开始恢复','{"intent":"run","case_name":null,"fields":{}}'),
    ('暂停采集','{"intent":"pause","case_name":null,"fields":{}}'),
]


def main():
    from mlx_lm import load,generate
    from mlx_lm.sample_utils import make_sampler
    model_dir=Path(__file__).parent/'assets/local-agent/qwen3-0.6b-4bit'
    model,tokenizer=load(str(model_dir))
    print(json.dumps(dict(ready=True,model='Qwen3-0.6B-4bit',backend='MLX')),flush=True)
    for line in sys.stdin:
        try:
            request=json.loads(line);started=time.perf_counter()
            context=request.get('context',{})
            messages=[{'role':'system','content':SYSTEM+'\nCurrent case: '+str(context.get('current_case'))+'; pending field: '+str(context.get('pending_field'))}]
            for user,answer in EXAMPLES:messages.extend([{'role':'user','content':user},{'role':'assistant','content':answer}])
            messages.append({'role':'user','content':request['user']})
            prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            text=generate(model,tokenizer,prompt=prompt,max_tokens=256,sampler=make_sampler(temp=0),verbose=False)
            # Invalid or out-of-schema output is rejected by the case layer.
            left=text.find('{');right=text.rfind('}')
            if left<0 or right<left:raise ValueError('The local model did not return JSON.')
            plan=json.loads(text[left:right+1])
            print(json.dumps(dict(plan=plan,seconds=time.perf_counter()-started),ensure_ascii=False),flush=True)
        except Exception as e:print(json.dumps(dict(error=str(e))),flush=True)


if __name__=='__main__':main()
