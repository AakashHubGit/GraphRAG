import os
import asyncio
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from rouge_score import rouge_scorer



load_dotenv()


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
    print(text)
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    return splitter.split_text(text)

def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index_sim")

# ---------------- RAG Chain ----------------
def get_rag_chain():
    template = """
    You are an AI assistant. Use the provided context from the document to answer the question.
    If the answer is not in the context, say "I couldn’t find that in the ducument."

    Context:
    {context}

    Question: {question}

    Answer:
    """
    llm=ChatGroq(model="openai/gpt-oss-120b")
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return  prompt | llm
    
   

def ask_question(question, k=3):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index_sim", embeddings, allow_dangerous_deserialization=True)

    # Retrieve relevant chunks
    docs = vector_store.similarity_search(question, k=k)
    context = " ".join([doc.page_content for doc in docs])

    chain = get_rag_chain()
    answer=chain.invoke({"context": context, "question": question}).content
    return answer

# ---------------- Run Example ----------------
if __name__ == "__main__":
    pdf_docs = [r"Docs.pdf"]
    if not os.path.exists("faiss_index_sim"):
        text = get_pdf_text(pdf_docs)
        chunks = get_text_chunks(text)
        create_vector_store(chunks)
        print("✅ Vector store created!")
    queries ="Which department offers the minor that is linked both to Sociology majors and non-Sociology majors, and who is the contact person responsible for it?"
    answers=ask_question(queries)
    print(answers)
    


# def evaluate_summary(generated_summary, reference_summary):
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
#     scores = scorer.score(reference_summary, generated_summary)
#     return scores

# # Example usage
# generated = answers
# reference = "The Department of Sociology offers the Minor in Crime, Law, and Justice Studies, which applies to both Sociology and non-Sociology majors. The contact person is Patricia Monture."

# scores = evaluate_summary(generated, reference)
# for metric, result in scores.items():
#     print(f"{metric}: Precision={result.precision:.4f}, Recall={result.recall:.4f}, F1={result.fmeasure:.4f}")




def evaluate_answer(question, generated_answer, reference_answer):
    judge_llm = ChatGroq(model="openai/gpt-oss-120b")
    prompt = f"""
    Question: {question}
    Generated Answer: {generated_answer}
    Reference Answer: {reference_answer}

    Evaluate the generated answer compared to the reference.
    Score on:
    - Accuracy (0–1)
    - Relevance (0–1)
    - Faithfulness to context (0–1)

    Return only a JSON like this:
    {{"accuracy": x, "relevance": y, "faithfulness": z}}
    """

    result = judge_llm.invoke(prompt)
    return result.content

reference_answer= "The Department of Sociology offers the Minor in Crime, Law, and Justice Studies, which applies to both Sociology and non-Sociology majors. The contact person is Patricia Monture."
score = evaluate_answer(queries,answers, reference_answer)
print("\n📊 Evaluation Score:", score)