class Trace:
    def __init__(self):
        self.events=[]

    def add(self,name,detail):
        self.events.append({'event':name,'detail':detail})

    def export(self):
        return self.events
