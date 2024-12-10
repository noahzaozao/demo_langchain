from langserve import RemoteRunnable  # Import the RemoteRunnable class from langserve

# URL of the remote service you want to connect to
url = "http://127.0.0.1:8000/chain_demo"

def get_chain_demo():
    # Create an instance of RemoteRunnable with the given URL
    client = RemoteRunnable(url)
    
    # Invoke the remote service with a dictionary of inputs (language and question)
    result = client.invoke({
        "language": "English",  # Specify the language of the input
        "question": "How heavy is a lion?"  # The question you want to ask the service
    })
    
    try:
        # Attempt to print the content returned by the service
        print(result.content)
    except:
        # If an error occurs, print the result directly
        print(result)

# Call the function to get the chain demo result
get_chain_demo()
