import os
import sys
import json

if len(sys.argv) < 2:
    print('Usage: python tools/video_inspect.py <video_path>')
    sys.exit(2)

video = sys.argv[1]
os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.getcwd())
from backend.server import analyze_video_local

try:
    r = analyze_video_local(video)
    print(json.dumps(r, indent=2))
except Exception as e:
    print(json.dumps({'error': str(e)}))
