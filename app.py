import transformers
from transformers import utils, pipeline, set_seed
import torch
from flask import Flask, request, render_template, session, redirect

app = Flask(__name__)

# Set the secret key for the session
app.secret_key = 'your-secret-key'

# Select only one model to use at a time by uncommenting it
# MODEL_NAME = "EleutherAI/gpt-j-6B"
MODEL_NAME = "facebook/opt-125m"
# MODEL_NAME = "facebook/opt-350m"
# MODEL_NAME = "facebook/opt-1.3b"
# MODEL_NAME = "facebook/opt-2.7b"
# The models below this point may not run on GPU's with 4GB or less memory
# MODEL_NAME = "facebook/opt-6.7b"
# MODEL_NAME = "facebook/opt-13b"
# MODEL_NAME = "facebook/opt-30b"
# MODEL_NAME = "facebook/opt-66b"
# MODEL_NAME = "facebook/opt-175b" # request access to model here: https://docs.google.com/forms/d/e/1FAIpQLSe4IP4N6JkCEMpCP-yY71dIUPHngVReuOmQKDEI1oHFUaVg7w/viewform

# Initialize the chat history
history = [""]

# Select only one generator function at a time by uncommenting it
# generator = pipeline("text-generation", model=f"{MODEL_NAME}", do_sample=True, torch_dtype=torch.half)
# generator = pipeline("text-generation", model=f"{MODEL_NAME}", do_sample=True, torch_dtype=torch.float32)
# generator = pipeline("text-generation", model=f"{MODEL_NAME}", do_sample=True, torch_dtype=torch.float64)
generator = pipeline("text-generation", model=f"{MODEL_NAME}", do_sample=True)

# Define the chatbot logic
def chatbot_response(input_text, history):
    # Concatenate the input text and history list
    input_text = "\n".join(history) + "\nUser: " + input_text + " ChatGPT: "
    set_seed(32)
    response_text = generator(input_text, max_length=1024, num_beams=1, num_return_sequences=1)[0]["generated_text"]
    # Extract the bot's response from the generated text
    response_text = response_text.split("ChatGPT:")[-1]
    # Cut off any "User:" or "User:" parts from the response
    response_text = response_text.split("User:")[0]
    response_text = response_text.split("User:")[0]
    return response_text

@app.route("/", methods=["GET", "POST"])
def index():
    global history  # Make the history variable global
    if request.method == "POST":
        input_text = request.form["input_text"]
        response_text = chatbot_response(input_text, history)
        # Append the input and response to the chat history
        history.append(f"User: {input_text}")
        history.append(f"ChatGPT: {response_text}")
    else:
        input_text = ""
        response_text = "How can I help you?\n"
        history.append(f"ChatGPT: {response_text}")
    # Render the template with the updated chat history
    return render_template("index.html", input_text=input_text, response_text=response_text, history=history)

@app.route("/reset", methods=["POST"])
def reset():
    global history  # Make the history variable global
    history = [""]
    # Redirect to the chat page
    return redirect("/")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
