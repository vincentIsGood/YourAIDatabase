# from types import FunctionType
from typing import Callable

class CommandHandler:
    def __init__(self, cmd: str = "", desc: str = ""):
        self.cmd = cmd
        self.desc = desc
        self.console = None

    def setConsole(self, console: 'InteractiveConsole'):
        self.console = console

    def handle(self, args: 'list[str]'):
        pass

    def helpString(self):
        return f"{self.cmd.rjust(self.console.longestCmdLen)}    {self.desc}"

class SimpleCommandHandler(CommandHandler):
    def __init__(self, handler: Callable, cmd: str = "", desc: str = ""):
        super().__init__(cmd, desc)
        self.handler = handler

    def handle(self, args: 'list[str]'):
        self.handler(args)

class InteractiveConsole:
    def __init__(self, prompt = "> "):
        self.prompt = prompt
        self.handlers: 'list[CommandHandler]' = []
        self.longestCmdLen = 0
        self.longestDescLen = 0

        self.addHandler(HelpCommandHandler("help", "help me!"))
        self.addHandler(CommandHandler("exit", ""))

    def findLongestProperties(self):
        longestCmd = 0
        longestDesc = 0
        for handler in self.handlers:
            cmdLength = len(handler.cmd)
            descLength = len(handler.desc)
            if cmdLength > longestCmd:
                longestCmd = cmdLength

            if descLength > longestDesc:
                longestDesc = descLength

        self.longestCmdLen = longestCmd
        self.longestDescLen = longestDesc

    def addHandler(self, handler: CommandHandler):
        handler.setConsole(self)
        self.handlers.append(handler)
        self.findLongestProperties()

    def takeover(self):
        if len(self.handlers) == 0:
            return
        
        while True:
            try:
                userInput = input(self.prompt).strip()
                if userInput == "":
                    continue

                if userInput == "exit":
                    break

                splited = userInput.split()
                for handler in self.handlers:
                    cmd = splited[0]
                    if cmd == handler.cmd:
                        handler.handle(splited[1:])
            except KeyboardInterrupt as _:
                print("[!] Interrupt detected. Closing application.")
                break
            except Exception as e:
                print("[-] Error encountered: %s" % str(e))

### Default Handlers
class HelpCommandHandler(CommandHandler):
    def __init__(self, cmd: str = "", desc: str = ""):
        super().__init__(cmd, desc)

    def handle(self, args):
        print(f"{'[Cmd]'.rjust(self.console.longestCmdLen)}    {'[Description]'}")
        for handler in self.console.handlers:
            print(handler.helpString())