from typing import Any, TextIO

from langchain.callbacks.base import BaseCallbackHandler

class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, outputStream: TextIO):
        super().__init__()
        self.outputStream = outputStream

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.outputStream.write(token)
        self.outputStream.flush()
