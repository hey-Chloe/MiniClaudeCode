class SandboxRuntime:
    def execute(self,command):
        return {'isolated':True,'command':command}
