import modal

app = modal.App("inference")

@app.function()
def subtract_five(x):
    print("This code is running on a remote worker!")
    return x - 5

@app.local_entrypoint()
def main():
    print("42 - 5 is", subtract_five.remote(42))
