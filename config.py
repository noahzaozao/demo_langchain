import os


def set_environment():
    proxy = '127.0.0.1:7890'
    os.environ.update({
        'http_proxy': proxy,
        'https_proxy': proxy,
        'LANGCHAIN_TRACING_V2': 'true',
        'LANGCHAIN_API_KEY': '',
        'LANGCHAIN_PROJECT': 'DemoLangchain',
        # 'OPENAI_API_KEY': '',
        'DASHSCOPE_API_KEY': ''  # API_KEY for Tongyi
    })


set_environment()
