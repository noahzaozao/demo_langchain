import os


def set_environment():
    proxy = '127.0.0.1:7890'
    os.environ.update({
        'http_proxy': proxy,
        'https_proxy': proxy,
        'OPENAI_API_KEY': '',
        'OPENAI_BASE_URL': 'https://api.bianxie.ai/v1'
    })

set_environment()
