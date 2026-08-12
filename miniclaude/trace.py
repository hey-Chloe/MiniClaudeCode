import json
import time
import uuid
from pathlib import Path


class Trace:
    def __init__(self):
        self.events=[]
        self.detailed_events=[]
        self.run_id=str(uuid.uuid4())

    def add(self,name,detail):
        self.events.append({'event':name,'detail':detail})
        self.detailed_events.append({
            'run_id': self.run_id,
            'step': len(self.detailed_events),
            'timestamp': time.time(),
            'event': name,
            'detail': detail,
        })

    def clear(self):
        self.events.clear()
        self.detailed_events.clear()
        self.run_id=str(uuid.uuid4())

    def export(self):
        return [event.copy() for event in self.events]

    def export_detailed(self):
        return [event.copy() for event in self.detailed_events]

    def write_jsonl(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rendered = ''.join(json.dumps(event, ensure_ascii=False, default=str) + '\n' for event in self.detailed_events)
        target.write_text(rendered, encoding='utf-8')
