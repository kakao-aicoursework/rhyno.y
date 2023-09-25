import os
from datetime import datetime

import pynecone as pc
from pynecone.base import Base
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions
from langchain.memory import FileChatMessageHistory

# TODO
# 1. llm_query 중에는 입력창 닫기
# 2. view / llm / model 파일 분리

# ------------------------------------------------------------------------------------------
# -------- Global variable
# ------------------------------------------------------------------------------------------
os.environ["OPENAI_API_KEY"] = open("api-key.txt").read()
DATA_BY_API = {
    "KAKAO_SYNC": "project_data_카카오싱크.txt",
    "KAKAO_SOCIAL": "project_data_카카오소셜.txt",
    "KAKAO_CHANNEL": "project_data_카카오채널.txt",
}
LLM = ChatOpenAI(temperature=0.8, max_tokens=500, model="gpt-3.5-turbo-16k")
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../.web/data')
CHROMA_CLIENT = chromadb.PersistentClient(path=DB_PATH, settings=Settings(allow_reset=True, anonymized_telemetry=False))
COLLECTION = CHROMA_CLIENT.get_or_create_collection(
    name="kakao_api",
    embedding_function=embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY")
    )
)
HISTORY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../.web/data')

SYSTEM_MESSAGE = f"""assistant 는 카카오 API 에 대한 챗봇 입니다.
카카오 API 는 <API종류> 에 제시된 {len(DATA_BY_API)} 가지 입니다.
<API종류>
{", ".join(list(DATA_BY_API.keys()))}
<API종류>
 
질문의 답은 간결하고 명료하게 markdown 문법의 목록으로 작성한다.

예시는 다음과 같다.
<예시>
{{질문에 대한 답 요약}}

1.회원가입
- {{회원가입에 대한 내용}}

{{공백}}

2.간편한 서비스약관 관리
- {{서비스약관 처리 간소화 내용}}
<예시>
"""
USER_TEMPLATE = """
다음에 제시된 사용자의 질문에 답하세요.

질문: {question}

<연관문서>
{related_documents}
<연관문서>
"""
ANSWER_CHAIN = LLMChain(llm=LLM, prompt=ChatPromptTemplate.from_messages(
    [SystemMessage(content=SYSTEM_MESSAGE),
     HumanMessagePromptTemplate.from_template(template=USER_TEMPLATE)]
))

# TODO: langchain 의 routing chain 으로 바꿀 것
API_TYPE_CHAIN = LLMChain(llm=LLM, prompt=ChatPromptTemplate.from_template(
    f"""
카카오 API 는 <API종류> 에 제시된 {len(DATA_BY_API)} 가지 입니다.
<API종류>
{", ".join(list(DATA_BY_API.keys()))},unknown
<API종류>

다음의 <질문>이 어떤 API 종류에 대한 것인지 구분하세요.

<질문>
{{question}}
<질문>

대답은 단답형으로 제시된 <API종류> 의 단어 중 1개만 포함합니다. 
다른 단어나 문장은 쓰지 않습니다.
unknown 은 API 종류를 특정할 수 없는 경우만 답변합니다.

ex)
unknown
"""
))


# ------------------------------------------------------------------------------------------
# -------- LLM Helper
# ------------------------------------------------------------------------------------------

def history_of_api_type(conversation_id: str):
    file_path = os.path.join(HISTORY_PATH, f"{conversation_id}.json")
    return FileChatMessageHistory(file_path)


def pop_api_type_in_history(conversation_id: str = "last_api_type"):
    history = history_of_api_type(conversation_id)
    if len(history.messages) == 0:
        return "unknown"
    else:
        return history.messages.pop().content


def push_api_type_to_history(api_type: str, conversation_id: str = "last_api_type"):
    history = history_of_api_type(conversation_id)
    history.add_user_message(api_type)


def llm_query(question: str):
    api_type = API_TYPE_CHAIN.run(question=question)

    if api_type == "unknown":
        print(f"Can not judge by user input. search from history")
        api_type = pop_api_type_in_history()
        print(f"API_TYPE_BY_HISTORY: {api_type}")
    else:
        push_api_type_to_history(api_type)

    print(f"API_TYPE_CHAIN's answer: {api_type}")
    if api_type == "unknown":
        # TODO: web search
        return f"""잘못된 질문입니다. 대답할 수 있는 API 종류는 {", ".join(list(DATA_BY_API.keys()))} 입니다."""
    elif api_type in DATA_BY_API:
        related_documents = COLLECTION.query(query_texts=[f"api: {api_type}" + question], n_results=3)
        print(f"related_documents from vectordb: {related_documents}")
        answer = ANSWER_CHAIN.run(api_type=api_type, question=question, related_documents=related_documents)
        print(f"ANSWER_CHAIN's answer: {answer}")
        return answer
    else:
        raise Exception("cannot be here")


# ------------------------------------------------------------------------------------------
# -------- User Define Structure
# ------------------------------------------------------------------------------------------

class QnA(Base):
    question: str
    answer: str
    created_at: str


# ------------------------------------------------------------------------------------------
# -------- Framework State Model
# ------------------------------------------------------------------------------------------

class State(pc.State):
    """The app state."""

    qna_list: list[QnA] = []
    is_working: bool = False

    # TODO 화면노출
    def validate_not_empty(self, value: str):
        if value.strip() == '':
            raise Exception("input empty")

    async def on_submit(self, form_data):
        question = form_data["question"]
        self.validate_not_empty(question)
        self.is_working = True
        qna = QnA(
            question=form_data["question"],
            answer="(ing) looking for an answer",
            created_at=datetime.now().strftime("%I:%M%p on %B %d, %Y")
        )
        self.qna_list.append(qna)
        yield
        answer = llm_query(question=question)
        print(answer)
        qna.answer = answer
        self.qna_list.remove(qna)
        self.is_working = False
        self.qna_list.append(qna)
        yield


# ------------------------------------------------------------------------------------------
# -------- Define views
# ------------------------------------------------------------------------------------------

def down_arrow():
    return pc.vstack(
        pc.icon(
            tag="arrow_down",
            color="#666",
        )
    )


def text_box(text):
    return pc.text(
        text,
        background_color="#fff",
        padding="1rem",
        border_radius="8px",
    )


def qna_view(qna):
    return pc.box(
        pc.vstack(
            text_box("Q. " + qna.question),
            down_arrow(),
            pc.box(
                pc.markdown(
                    qna.answer
                ),
                margin="5rem",
                font_size="0.8rem",
                padding="1rem",
                background_color="white"
            ),
            spacing="0.3rem",
            align_items="left",
        ),
        background_color="#f5f5f5",
        padding="1rem",
        border_radius="8px",
    )


def index():
    """The main view."""
    return pc.container(
        # header
        pc.box(
            pc.hstack(
                pc.text("ChatBot 🤖", font_size="2rem"),
                pc.cond(State.is_working,
                        pc.spinner(
                            color="red",
                            thickness=7,
                            speed="3s",
                            size="l",
                        ),
                        ),
            ),
            pc.text(
                "Answer about KAKAO API",
                margin_top="0.5rem",
                color="#666",
            ),
        ),
        pc.form(
            pc.vstack(
                pc.input(
                    placeholder="> Question",
                    margin_top="1rem",
                    border_color="#eaeaef",
                    id="question"
                ),
                pc.button("💬 send", type_="submit", margin_top="1rem")
            ),
            on_submit=State.on_submit,
        ),
        pc.vstack(
            pc.foreach(State.qna_list, qna_view),
            margin_top="2rem",
            spacing="1rem",
            align_items="left"
        ),
        padding="2rem",
        min_width="600px"
    )


# ------------------------------------------------------------------------------------------
# -------- Add state and page to the app.
# ------------------------------------------------------------------------------------------

def init():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        add_start_index=True,
    )
    id = 1
    for apiType, fileName in DATA_BY_API.items():
        with open(fileName) as f:
            content = f.read()

            ids = []
            metas = []
            documents = []
            texts = text_splitter.create_documents([content])
            text_len = len(texts)
            for i in range(0, text_len):
                ids.append(str(id))
                metas.append({'apiType': apiType})
                documents.append(texts[i].page_content)
                id += 1

            COLLECTION.add(ids=ids, metadatas=metas, documents=documents)


init()
app = pc.App(state=State)
app.add_page(index, title="KAKAO-API QnA")
app.compile()
