"""Deterministic case commands, independent of the local language model."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from casework import CaseStore,ALLOWED,HERE

HELP = '''/case new NAME                   Create and select a case
/case open NAME                  Select an existing case
/case list                       List saved cases
/case show [NAME]                Show metadata
/case set [NAME] FIELD VALUE      Set a metadata field
/case import [NAME] FILE          Import a JSON object of metadata fields
/case check [NAME]               Validate evidence and show missing information
/case plan [NAME]                Preview registered recovery arguments
/case run [NAME]                 Request recovery with the current templates

Fields: model_name, observation, methods, calibration, notes, max_positions, max_queries.
Methods use comma-separated IDs: guided,collision_plus,exhaustive.
Use unknown for an unidentified model, or null to clear an optional field.
For shell scripts, use ./casectl with the same subcommands and explicit case names.'''


class CaseCommands:
    def __init__(self,store=None):self.store=store or CaseStore();self.current=None

    def value(self,field,parts):
        raw=' '.join(parts)
        if field not in ALLOWED:raise ValueError('Unknown field. '+', '.join(sorted(ALLOWED)))
        if field=='notes':return raw
        if raw=='null':return None
        if field in ('max_queries','max_positions'):return int(raw)
        if field=='methods':return [x.strip() for x in raw.split(',')]
        return raw

    def execute(self,args):
        action=args[0] if args else 'help';values=args[1:];intent='info';case=None
        if action=='help':return dict(reply=HELP,intent='info',origin='command')
        if action=='list':
            names=sorted(p.parent.name for p in self.store.root.glob('*/metadata.json'))
            return dict(reply='\n'.join(names) or 'No cases yet. Use /case new NAME.',intent='info',origin='command')
        if action in ('new','open'):
            if len(values)!=1:raise ValueError(f'/case {action} NAME')
            name=values[0]
            case=self.store.create(name) if action=='new' else self.store.load(name)
        elif action=='set':
            if values and values[0] in ALLOWED:
                name=self.current;parts=values
            else:
                name=values[0] if values else None;parts=values[1:]
            if not name or len(parts)<2:raise ValueError('/case set [NAME] FIELD VALUE; select a case first or specify its name.')
            case=self.store.update(name,{parts[0]:self.value(parts[0],parts[1:])},source='fixed command')
        elif action=='import':
            if len(values)==1:name=self.current;file=values[0]
            elif len(values)==2:name,file=values
            else:raise ValueError('/case import [NAME] FILE')
            if not name:raise ValueError('Select a case first or specify its name.')
            fields=json.loads(Path(file).expanduser().read_text())
            case=self.store.update(name,fields,source='JSON import: '+str(Path(file).expanduser().resolve()))
        elif action in ('show','check','plan','run'):
            if len(values)>1:raise ValueError(f'/case {action} [NAME]')
            name=values[0] if values else self.current
            if not name:raise ValueError('Select a case first or specify its name.')
            if action in ('plan','run'):
                case,plans=self.store.plan(name)
                if action=='plan':
                    reply=json.dumps(plans,ensure_ascii=False,indent=2)
                else:reply='Recovery requested. Checking collection status and cooling before execution.';intent='run'
            else:case=self.store.load(name) if action=='show' else self.store.inspect(name)
        else:raise ValueError('Unknown case subcommand. Use /case help.')
        self.current=case['name']
        if action=='show':reply=json.dumps(case,ensure_ascii=False,indent=2)
        elif action not in ('plan','run'):
            if action!='check':case=self.store.inspect(self.current)
            reply=f"Selected {self.current}. "+self.store.question(case)
        self.store.record(self.current,'user','/case '+' '.join(args),source='fixed command')
        self.store.record(self.current,'assistant',reply,source='fixed command')
        return dict(reply=reply,intent=intent,origin='command',case=self.current,
                    model_name=case['fields'].get('model_name'),ready=case.get('readiness',{}).get('ready',False),
                    metadata=str(self.store.path(self.current)/'metadata.json'))


def main():
    parser=argparse.ArgumentParser(description='Fixed case metadata commands. No language model is loaded.',epilog=HELP)
    parser.add_argument('arguments',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    try:
        result=CaseCommands().execute(args.arguments);print(result['reply'],flush=True)
        if result['intent']=='run':
            from casework import ROOT
            python=HERE.parent/'experiments/kv_guided_search_gemma4_int8/.venv/bin/python'
            output=CaseStore().path(result['case'])/'runs'/('run_'+str(time.time_ns()))
            sys.exit(subprocess.call(['/usr/bin/caffeinate','-i','-s',str(python),str(HERE/'run_case.py'),result['case'],'--run-dir',str(output)],cwd=ROOT))
    except (ValueError,OSError,KeyError) as e:parser.exit(1,str(e)+'\n')


if __name__=='__main__':main()
