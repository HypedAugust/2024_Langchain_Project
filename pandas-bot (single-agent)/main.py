import os
import pandas as pd
import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

# Set the matplotlib backend to 'Agg' to disable interactive mode
matplotlib.use('Agg')

# Set OpenAI API key
OPEN_API_KEY = os.getenv("OPEN_AI_KEY")

# Configure the LLM model
llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0, api_key=OPEN_API_KEY)

def analyze_with_langchain_agent(df, question):
    agent_executor = create_pandas_dataframe_agent(
        llm,
        df,
        agent_type='openai-tools',
        verbose=True,
        return_intermediate_steps=True,
        allow_dangerous_code=True
    )
    response = agent_executor.invoke(question)
    text_output = response['output']
    python_code = None

    try:
        for item in response['intermediate_steps']:
            if item[0].tool == 'python_repl_ast':
                python_code = str(item[0].tool_input['query'])
    except:
        pass

    return text_output, python_code

def execute_and_show_chart(python_code, df):
    try:
        locals = {'df': df.copy()}
        exec(python_code, globals(), locals)

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        return img
    except Exception as e:
        print(f'Error executing chart code: {e}')
        return None

def process_and_display(csv_file, question):
    df = pd.read_csv(csv_file)
    text_output, python_code = analyze_with_langchain_agent(df, question)
    chart_image = execute_and_show_chart(python_code, df) if python_code else None
    return text_output, chart_image

with gr.Blocks() as demo:
    gr.Markdown("파일을 업로드 후, 질문입력")
    with gr.Row():
        csv_input = gr.File(label="CSV 파일 업로드", type="filepath")
        input = gr.Textbox(label="질문을 입력하세요.")
        btn = gr.Button("process_and_display")
    
    output_markdown = gr.Markdown('마크다운 형태 결과')
    output_image = gr.Image()

    btn.click(fn=process_and_display, inputs=[csv_input, input], outputs=[output_markdown, output_image])

demo.launch(share=True)
