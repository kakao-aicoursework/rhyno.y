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

# TODO
# 1. llm_query 중에는 입력창 닫기
# 2. view / llm / model 파일 분리

# ------------------------------------------------------------------------------------------
# -------- Global variable
# ------------------------------------------------------------------------------------------
os.environ["OPENAI_API_KEY"] = open("api-key.txt").read()
data_by_api = {
    "KAKAO_SYNC": open("project_data_카카오싱크.txt").read(),
}
chat = ChatOpenAI(temperature=0.8)


# ------------------------------------------------------------------------------------------
# -------- LLM Helper
# ------------------------------------------------------------------------------------------

def llm_query(question: str, api_type: str):
    system_message = """assistant는 {api_type} 에 대한 chatbot 입니다.
    다음 <{api_type}> 안에 제시되는 {api_type} 정보를 토대로 user 의 질문에 답한다.
<{api_type}>
{data_by_api}
<{api_type}>

질문의 답은 간결하고 명료하게 markdown 문법의 목록으로 작성한다.

예시는 다음과 같다.
<예시>
{질문에 대한 답 요약}

1.회원가입
- {회원가입에 대한 내용}

{공백줄}

2.간편한 서비스약관 관리
- {서비스약관 처리 간소화 내용}
<예시>
"""
    system_message_prompt = SystemMessage(content=system_message)

    human_template = ("질문: {question}")

    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    return chain.run(api_type=api_type, question=question)


# ------------------------------------------------------------------------------------------
# -------- User Define Structure
# ------------------------------------------------------------------------------------------

class QnA(Base):
    api_type: str
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
        api_type = form_data["api_type"]
        question = form_data["question"]
        self.validate_not_empty(api_type)
        self.validate_not_empty(question)
        self.is_working = True
        qna = QnA(
            api_type=form_data["api_type"],
            question=form_data["question"],
            answer="(ing) looking for an answer",
            created_at=datetime.now().strftime("%I:%M%p on %B %d, %Y")
        )
        self.qna_list.append(qna)
        yield
        answer = llm_query(api_type=api_type, question=question)
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
            pc.box(
                pc.text(qna.api_type),
                pc.text(" · ", margin_x="0.3rem"),
                pc.text(qna.created_at),
                display="flex",
                font_size="0.8rem",
                color="#666",
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
                pc.select(
                    list(data_by_api.keys()),
                    default_value=list(data_by_api.keys())[0],
                    placeholder="> Choose KAKAO-API",
                    margin_top="1rem",
                    id="api_type"
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

app = pc.App(state=State)
app.add_page(index, title="KAKAO-API QnA")
app.compile()
