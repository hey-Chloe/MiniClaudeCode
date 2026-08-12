class ToolRegistry:
    def __init__(self):
        self.tools={}

    def register(self,tool):
        self.tools[tool.name]=tool

    def list_tools(self):
        return list(self.tools.keys())
