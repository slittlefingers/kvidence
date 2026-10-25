"""Natural-language case orchestration with a small local model and validated templates."""
import json
import os
import re
from pathlib import Path
import selectors
import subprocess
import threading
from casework import CaseStore,HERE

PY=HERE.parent/'experiments/kv_guided_search_gemma4_int8/.venv/bin/python'


class LocalModel:
    def __init__(self):self.process=None;self.lock=threading.Lock()
    def receive(self,timeout=90):
        selector=selectors.DefaultSelector()
        try:
            selector.register(self.process.stdout,selectors.EVENT_READ)
            if not selector.select(timeout):raise TimeoutError('Local agent timed out. No case action was executed.')
            line=self.process.stdout.readline()
            if not line:raise RuntimeError('Local agent stopped. Check assets/local-agent/worker.log.')
            return json.loads(line)
        finally:selector.close()
    def infer(self,text,context):
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                folder=HERE/'assets/local-agent';folder.mkdir(parents=True,exist_ok=True)
                env=dict(os.environ,HF_HUB_OFFLINE='1')
                with (folder/'worker.log').open('a') as log:
                    self.process=subprocess.Popen([str(PY),'-u',str(HERE/'local_agent_worker.py')],
                        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=env)
                greeting=self.receive()
                if not greeting.get('ready'):raise RuntimeError(str(greeting))
            self.process.stdin.write(json.dumps(dict(user=text,context=context),ensure_ascii=False)+'\n');self.process.stdin.flush()
            response=self.receive()
            if 'error' in response:raise ValueError(response['error'])
            return response
    def close(self):
        if self.process and self.process.poll() is None:self.process.terminate()


class CaseAgent:
    def __init__(self,store=None,model=None):
        self.store=store or CaseStore();self.model=model or LocalModel();self.current=None;self.lock=threading.Lock();self.awaiting_name=False
    def handle(self,text):
        with self.lock:
            case=self.store.inspect(self.current) if self.current else None
            missing=case['readiness']['missing'] if case else []
            context=dict(current_case=self.current,pending_field='case_name' if self.awaiting_name else missing[0] if missing else None,
                         fields=case['fields'] if case else {})
            response=self.model.infer(text,context);plan=response['plan']
            if not isinstance(plan,dict) or set(plan)-{'intent','case_name','fields'}:raise ValueError('Local agent returned an unsupported plan.')
            if self.awaiting_name and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',text.strip()):
                plan=dict(intent='create',case_name=text.strip(),fields={})
            intent=plan.get('intent');proposed=plan.get('case_name')
            explicit_name=proposed if isinstance(proposed,str) and proposed in text else None
            name=explicit_name if intent=='create' else explicit_name or self.current
            fields=plan.get('fields',{})
            if intent not in ('create','update','run','status','methods','pause','resume','help'):raise ValueError('Unknown agent intent; no action executed.')
            if not isinstance(fields,dict):raise ValueError('The agent did not return a metadata object.')
            # A tiny model can confuse choosing methods with authorizing execution.
            if intent=='run' and (not re.search(r'开始|启动|执行|运行|恢复一下|跑一下|跑一遍|\bstart\b|\brun\b|go ahead|proceed',text,re.I) or re.search(r'不要|暂不|先不|别运行|do not|don.t|not now',text,re.I)):intent='update'
            if intent=='pause' and not re.search(r'暂停|停止采集|\bpause\b|stop collection',text,re.I):intent='update'
            if intent=='resume' and not re.search(r'继续采集|恢复采集|\bresume\b|start collection',text,re.I):intent='update'
            if re.search(r'不知道|不确定|不清楚|unknown|not sure|don.t know',text,re.I) and re.search(r'模型|model',text,re.I):
                fields['model_name']='unknown'
            if fields.get('model_name') in ('gemma4-e2b','gemma3-1b'):
                number='4' if fields['model_name']=='gemma4-e2b' else '3'
                if not re.search(r'gemma\s*'+number,text,re.I):fields.pop('model_name')
            # Normalize explicit domain terms; the model is not trusted to spell tool IDs.
            selected=[]
            if re.search(r'岭回归|\bridge\b|\bguided\b',text,re.I):selected.append('guided')
            if re.search(r'collision|碰撞',text,re.I):
                selected.append('collision_plus_cpa' if re.search(r'official|\bcpa\b|官方|原版',text,re.I) else 'collision_plus')
            if re.search(r'穷举|\bexhaustive\b|brute.?force',text,re.I):selected.append('exhaustive')
            if selected:fields['methods']=selected
            paths=re.findall(r'(?:~?/|\.\.?/|kvidence/)[^\s\"\'，。]+',text)
            if paths:
                pending=context['pending_field']
                field='calibration' if re.search(r'calibration|校准',text,re.I) or pending=='calibration' else 'observation'
                if len(paths)==1:fields[field]=paths[0]
            for field in ('observation','calibration'):
                if fields.get(field) and fields[field] not in text:raise ValueError('The model proposed a path that was not supplied. Please paste the exact local path.')
            if intent=='create':
                if not name:
                    self.awaiting_name=True
                    return dict(reply='What should I call the case? For example: case1.',intent='question')
                self.store.create(name);self.current=name;self.awaiting_name=False
            elif name:
                self.store.load(name);self.current=name
            elif intent not in ('help','methods','pause','resume'):
                return dict(reply='Create a case first, for example: create case1.',intent='question')
            if self.current:
                self.store.record(self.current,'user',text,local_model='Qwen3-0.6B-4bit',model_plan=plan,inference_seconds=response.get('seconds'))
                if fields:self.store.update(self.current,fields,source='user text extracted by local model')
                case=self.store.inspect(self.current)
            if intent=='help':reply='Describe a case, provide the requested information, then say start recovery. I use registered templates and record the conversation in the case metadata directory.'
            elif intent=='methods':reply='Ridge-guided, Collision+ shared-RMSE adaptation, and exhaustive token-ID enumeration are available. Official Collision+ CPA is also available with independent calibration.'
            elif intent in ('pause','resume'):reply=f'Collection {intent} requested.'
            elif intent=='run' and case['readiness']['ready']:reply=f"Recovery requested for {self.current}. Checking collection state before launching the registered methods."
            else:reply=self.store.question(case)
            if self.current:self.store.record(self.current,'assistant',reply)
            return dict(reply=reply,intent=intent,case=self.current,ready=case['readiness']['ready'] if case else False,
                        model_name=case['fields'].get('model_name') if case else None,
                        metadata=str(self.store.path(self.current)/'metadata.json') if self.current else None)
