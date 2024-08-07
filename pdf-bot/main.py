# 서버에서 런칭하는법 : poetry run python main.py
# 환경 변수에서 API키 가져오기 
import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# langchain 패키지 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import gradio as gr

# RAG Chain 구현 패키지 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chroma 임포트 방식 변경
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# gradio 인터페이스를 위한 패키지 
from gradio_pdf import PDF

# PDF 파일을 읽어서 벡터 저장소에 저장
def load_pdf_to_vector_store(pdf_file, chunk_size=1000, chunk_overlap=100, similarity_metric='cosine'):
    # pdf 파일 로딩
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()

    # 문서 청킹 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(documents)

    # Chroma 인스턴스 생성 및 문서 임베딩으로 초기화 
    embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embedding,
        collection_name="pdf_collection"  # 컬렉션 이름 지정
    )

    return vectorstore

# 벡터 저장소에서 문서를 검색하고 답변을 생성 
def retrieve_and_generate_answers(vectorstore, message, temperature=0):

    # RAG 체인 생성
    retriever = vectorstore.as_retriever()

    # prompt
    template = '''Answer the question based only on the following context:
    <context>
    {context}
    </context>

    Question: {input}
    '''

    prompt = ChatPromptTemplate.from_template(template)

    # Chatmodel 인스턴스 생성 
    model = ChatOpenAI(model='gpt-3.5-turbo-0125', temperature = temperature , api_key = OPENAI_API_KEY)

    # prompt와 chatmodel을 chain으로 연결 
    document_chain = create_stuff_documents_chain(model, prompt)

    # retriever를 chain에 연결 
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    # 검색 결과를 바탕으로 답변 생성 
    response = rag_chain.invoke({'input':message}) 

    return response['answer']


# Gradio 인터페이스에서 사용할 함수 
def process_pdf_and_answer(message, history, pdf_file, chunk_size, chunk_overlap, similarity_metric, temperature):

    # pdf 파일을 읽어서 벡터 저장소에 저장 
    vectorstore = load_pdf_to_vector_store(pdf_file, chunk_size, chunk_overlap, similarity_metric)

    answer = retrieve_and_generate_answers(vectorstore, message, temperature)

    return answer

demo = gr.ChatInterface(fn = process_pdf_and_answer, 
                        additional_inputs=[
                            PDF(label="PDF를 업로드 하세요."),
                            gr.Number(label="Chunk Size",value=1000),
                            gr.Number(label="Chunk Overlap",value=200),
                            gr.Dropdown(["cosign","유클리드 거리(ld)"],label = "similarity metric",value='cosine',info="옵션 업데이트 예정!"),
                            gr.Slider(label="temperature", minimum=0, maximum=2, step=0.1, value=0)
                            ]
                        )

# 샘플 테스트 
demo.launch()