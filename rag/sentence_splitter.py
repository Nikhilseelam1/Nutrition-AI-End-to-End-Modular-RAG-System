
import spacy


def add_sentences_to_pages(pages_and_texts):
    nlp = spacy.load("en_core_web_sm")

    for item in pages_and_texts:
        doc = nlp(item["text"])
        item["sentences"] = [str(sent) for sent in doc.sents]

    return pages_and_texts
