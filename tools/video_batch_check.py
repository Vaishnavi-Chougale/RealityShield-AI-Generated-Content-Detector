import os
import sys
import json

os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.getcwd())
from backend.server import analyze_video_local

video_dir = 'data/raw/videos'
files = sorted([f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4','.mov','.avi'))])

mismatches = []
for f in files:
    path = os.path.join(video_dir, f)
    try:
        r = analyze_video_local(path)
        predicted = r.get('verdict')
    except Exception as e:
        predicted = f'ERROR: {e}'
    actual = 'REAL' if f.lower().startswith('real_') else 'FAKE' if f.lower().startswith('fake_') else 'UNKNOWN'
    if predicted != actual:
        mismatches.append({'file': f, 'actual': actual, 'predicted': predicted})
    print(json.dumps({'file': f, 'actual': actual, 'predicted': predicted}))

print('\nMISMATCHES:\n')
print(json.dumps(mismatches, indent=2))
