import os
import sys
import json

os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.getcwd())
from backend.server import analyze_video_local

paths = [
    'data/raw/videos/real_video1.mp4',
    'data/raw/videos/fake_video1.mp4',
]

for p in paths:
    try:
        r = analyze_video_local(p)
        fa = len(r.get('frame_analysis', []))
        fc = sum(1 for f in r.get('frame_analysis', []) if f.get('label') == 'FAKE')
        out = {
            'path': p,
            'verdict': r.get('verdict'),
            'confidence': r.get('confidence'),
            'fake_probability': r.get('fake_probability'),
            'frame_count': fa,
            'fake_frames': fc,
        }
    except Exception as e:
        out = {'path': p, 'error': str(e)}
    print(json.dumps(out))
