"""Read Android thermal telemetry; never alter firmware thermal controls."""
import json,re,subprocess,time
from pathlib import Path
ADB=Path.home()/'Library/Android/sdk/platform-tools/adb'
class CoolingRequired(RuntimeError):pass

def thermal_snapshot(output):
 output=Path(output);output.mkdir(parents=True,exist_ok=True)
 try:r=subprocess.run([str(ADB),'-s','YOUR_DEVICE_SERIAL','shell','dumpsys','thermalservice'],capture_output=True,text=True,timeout=20)
 except subprocess.TimeoutExpired as e:raise CoolingRequired('Thermal telemetry timed out') from e
 match=re.search(r'Thermal Status:\s*(\d+)',r.stdout)
 if r.returncode or not match:raise CoolingRequired('Thermal telemetry unavailable')
 text=r.stdout.split('Current temperatures from HAL:')[-1].split('Current cooling devices')[0]
 sensors=[]
 for value,kind,name,status in re.findall(r'Temperature\{mValue=([^,]+), mType=(-?\d+), mName=([^,]+), mStatus=(\d+)\}',text):
  value=float(value);kind=int(kind)
  if kind in [0,1,2,3,9] and -20<=value<=150:sensors.append(dict(name=name,type=kind,celsius=value,status=int(status)))
 d=dict(unix_time=time.time(),thermal_status=int(match[1]),sensors=sensors)
 tmp=output/'thermal_latest.tmp';tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(output/'thermal_latest.json')
 with (output/'thermal_history.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')
 return d

def require_not_severe(output):
 d=thermal_snapshot(output)
 if d['thermal_status']>=3:raise CoolingRequired(f"Android severe thermal status {d['thermal_status']}")
 return d
