from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from lib.utils.InteractiveConsole import InteractiveConsole, SimpleCommandHandler
from lib.AiDatabase import AiDatabase

def main():
    aiDatabaseQuerier = AiDatabase([StreamingStdOutCallbackHandler()])

    def query(args):
        aiDatabaseQuerier.query(input("[*] Type in your query: "))

    def reload_qa(args):
        aiDatabaseQuerier.reloadPrompt()

    console = InteractiveConsole("(cmd)> ")
    console.addHandler(SimpleCommandHandler(query, "query", "Query your AI database"))
    console.addHandler(SimpleCommandHandler(reload_qa, "reload_qa", "Reload your prompt"))
    console.takeover()

main()