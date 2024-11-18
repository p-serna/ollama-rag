from ollama_rag import OllamaRAG



def parse_text(filename):
    with open(filename, 'r') as f:
        data = f.readlines()
    
    book = {}
    chapter = 0
    book[chapter] = []
    for i, line in enumerate(data):

        if line.strip().startswith(""):
            line = line.replace("", "").strip()
        if line.lower().strip().startswith("chapter"):
            chapter = line.strip()
            n_chapter = chapter.lower().split("chapter")[-1].strip()
            book[chapter] = []
        else:
            if line.strip() and  line.strip()[0].isdigit():
                text = re.sub("¶", "", line.strip())
                if text.upper() == text:
                    continue
                book[chapter].append(n_chapter+":"+text)
    return book



# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    rag = OllamaRAG(collection_name="bible")

    book_folder = "bible"
    filepaths = {filename.split(".txt")[0]: f"{book_folder}/{filename}" for filename in os.listdir(book_folder) if filename.endswith(".txt")}
    #filepaths.pop("Dedicatory")
    #filepaths.pop("Preface")
    books = list(filepaths.keys())
    #for book, filepath in filepaths.items():
    for book in books[58:]:
        filepath = filepaths[book]
        print(book)
        print(filepath)
        
        try:
            documents = [verse for verses in parse_text(filepath).values() for verse in verses]
        
            # Add documents
            rag.add_documents(
                texts=documents,
                ids=[f"{book} "+verse.split(" ")[0] for verse in documents],
                metadatas=[{"book": book, "verse": verse.split(" ")[0]}  for verse in documents]
            )
        except Exception as error:
            print(f"Error with {book}: {error}")

    
    # Example query
    question = "When was Eve created?"
    print(f"Question: {question}")

    answer = rag.query(question)
    
    # print(f"Answer: {answer}")
    print("Answer:")
    # for line in answer:
    #     print(f"{line}", end='')
