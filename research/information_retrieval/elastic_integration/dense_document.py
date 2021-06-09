import json

class DenseDocument:
    title: str
    body: str
    chunks: list
    def __init__(self, title: str, body:str, chunks:list):
        self.title = title
        self.body = body
        self.chunks = chunks

    def to_dict(self):
        return {'title': self.title, 'body': self.body, 'chunks': self.chunks}

    def __repr__(self):
        pretty = json.dumps(self.to_dict())
        return pretty
