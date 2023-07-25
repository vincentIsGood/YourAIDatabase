from flask import Flask, request
from markupsafe import escape, Markup
from ..lib.AiDatabase import AiDatabaseQuerier

app = Flask(__name__)
aiDatabaseQuerier = AiDatabaseQuerier()

@app.route("/gpt", methods=["GET"])
def handleDatabaseQuery():
    query = request.args.get("query")
    if query:
        aiDatabaseQuerier.query(Markup(query).unescape())
        return escape(generatedText)

