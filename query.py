import threading
from typing import Any
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from lib.utils.InteractiveConsole import InteractiveConsole, SimpleCommandHandler
from lib.AiDatabase import AiDatabase

queryJob = None

def main():
    aiDatabaseQuerier = AiDatabase([StreamingStdOutCallbackHandler()])

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