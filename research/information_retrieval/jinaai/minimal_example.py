from jina import Flow
from docarray import Document, DocumentArray

# A DocumentArray is a list of Documents.
docs = DocumentArray(
    [
        # We set the `text` of each Document to a line from "Squid Game"
        Document(
            text="I’m Good At Everything, Except The Things I Can’t Do."),
        Document(
            text="I Don’t Have A Home To Go Back To. In Here, I Stand A Chance At Least. But Out There? I Got Nothing Out There."
        ),
        Document(
            text="This Is Hell. What Are The Rules In Hell?"),
        Document(
            text="Do You Know What Someone With No Money Has In Common With Someone With Too Much Money? Living Is No Fun For Them."
        ),
        Document(
            text="You Don’t Trust People Here Because You Can. You Do It Because You Don’t Have Anybody Else."
        ),
    ]
)

# Create a new Flow to process our Documents
flow = (
    Flow()
    # Add encoder, to convert text to vector embeddings
    .add(uses="jinahub://TransformerTorchEncoder", name="encoder", install_requirements=True)
    # Add indexer
    # When indexing it embeds embeddings in a graph
    # When searching it retrieves nearest neighbor to search term
    .add(uses="jinahub://SimpleIndexer/v0.15", install_requirements=True, name="indexer")
)
# Open Flow as context manager
with flow:
    # Index our DocumentArray of Squid Game Documents
    flow.index(inputs=docs)
    # Create a Document containing our search term. In this case, we take it from user's input
    query = Document(text="Should I trust people here?")
    # Search the index, return similar matches, and store in `response`
    docs = flow.search(inputs=query)

# Pull out the matches from all the other data in the response
matches = docs[0].matches

print("Your search results")
print("-------------------\n")

for match in matches:
    # Print the text of each match (from `Document.text`)
    print(f"- {match.text}")