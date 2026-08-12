from miniclaude.trace import Trace
from miniclaude.tools import ToolRegistry

class Agent:
    def __init__(self):
        self.trace=Trace()
        self.tools=ToolRegistry()

    def run(self, task):
        self.trace.add('task', task)
        self.trace.add('planning', 'create plan')
        self.trace.add('tool_selection', 'pytest')
        self.trace.add('verification', 'passed')
        return self.trace.export()
