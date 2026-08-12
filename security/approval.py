class ApprovalManager:
    def check(self,command):
        return any(x in command for x in ['rm','docker','git commit'])
