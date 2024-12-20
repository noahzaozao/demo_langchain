import os

def set_environment():
    proxy = '127.0.0.1:7890'
    os.environ.update({
        'http_proxy': f'http://{proxy}',
        # 'https_proxy': proxy,
        'GOOGLE_API_KEY': '',
    })

set_environment()
