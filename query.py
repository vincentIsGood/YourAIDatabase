import sys
import threading
from typing import Any
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.callbacks.streaming_stdout import BaseCallbackHandler

from lib.utils.InteractiveConsole import InteractiveConsole, SimpleCommandHandler
from lib.AiDatabase import AiDatabase

queryJob = None

class StreamingCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, 
                     serialized, 
                     prompts, *, 
                     run_id, 
                     parent_run_id = None, 
                     tags = None, 
                     metadata = None, 
                     **kwargs) -> Any:
        for prompt in prompts:
            print(prompt)

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        sys.stdout.write(token)
        sys.stdout.flush()

def main():
    aiDatabaseQuerier = AiDatabase([StreamingCallbackHandler()])

    def query(args):
        global queryJob
        if queryJob and not queryJob.is_alive():
            queryJob = None
        if queryJob:
            print("[!] Still running, stop the model with 'stop'")
            return
        queryJob = threading.Thread(target=lambda: aiDatabaseQuerier.query(input("[*] Type in your query: ")), daemon=True)
        queryJob.start()

    def reload_qa(args):
        aiDatabaseQuerier.reloadPrompt()

    def stopModel(args):
        aiDatabaseQuerier.stopLLM()

    console = InteractiveConsole("(cmd)> ")
    console.addHandler(SimpleCommandHandler(query, "query", "Query your AI database"))
    console.addHandler(SimpleCommandHandler(reload_qa, "reload_qa", "Reload your prompt"))
    console.addHandler(SimpleCommandHandler(stopModel, "stop", "Reload your prompt"))
    console.takeover()

main()