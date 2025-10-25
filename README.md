# Local Command-Line Chatbot (ATG Technical Assignment)

This project is a fully functional, local command-line chatbot developed in Python for the ATG Machine Learning Intern technical assignment. It leverages the Hugging Face `transformers` library to run a conversational AI model (`microsoft/DialoGPT-medium`) entirely on your local machine.

The chatbot's core feature is its ability to maintain conversational context through a sliding window memory, allowing for coherent, multi-turn dialogues. The code is highly modular, separating concerns into model loading, memory management, and user interface logic, as specified by the assignment brief.

## Features

* **100% Local Execution**: Runs entirely on-device (CPU or GPU) without external API calls.
* **Conversational Memory**: Implements a sliding window buffer to remember the last 5 conversational turns, enabling context-aware follow-up questions.
* **Modular Architecture**: Code is logically organized into three distinct modules for maintainability:
    * `model_loader.py`: Handles all Hugging Face model and tokenizer loading.
    * `chat_memory.py`: A dedicated class for managing the conversation history `deque`.
    * `interface.py`: The main application loop, handling user I/O and state management.
* **Robust CLI Interface**: A clean, interactive command-line interface with `colorama` for improved readability.
* **Special Commands**:
    * `/exit`: Gracefully terminates the chatbot session.
    * `/clear`: Resets the chatbot's memory, allowing for a fresh start.

## Technical Deep-Dive & Design Decisions

This project was built with efficiency and modularity in mind, adhering to the assignment's technical specifications.

* **Model Selection**: `microsoft/DialoGPT-medium` was chosen as it provides a strong balance between conversational competence and resource requirements. It's powerful enough to hold a plausible conversation while still being small enough to run effectively on a standard CPU.
* **Memory Management**: The `ChatMemory` class uses Python's `collections.deque` with a `maxlen=5`. This is a highly efficient and Pythonic data structure for implementing a sliding window. When the 6th turn is added, the 1st turn is automatically and performantly discarded, ensuring the context prompt never exceeds the desired size.
* **Prompt Engineering**: Context is maintained by reconstructing the conversation history from memory and prepending it to the new user input, formatted as a continuous script (e.g., `User: ...\nBot: ...\nUser: ...\nBot:`). This is the format `DialoGPT` was trained on, leading to more accurate and in-context responses.
* **Response Sanitization**: A `strip_bot_text` helper function is used to parse the model's raw output. It cleanly extracts the reply, removes conversational artifacts (like the model hallucinating a new `User:` or `Bot:` prompt), and limits the response to the first line to prevent rambling.
* **Device Agnosticism**: The `model_loader.py` includes a `get_device` function to auto-detect a CUDA-enabled GPU (returning `0`) or default to CPU (returning `-1`). For this assignment, the device is explicitly set to CPU in `interface.py` to guarantee compatibility as per the "GPU optional" requirement.

## Project Structure
```
chatbot_project/
  interface.py
  model_loader.py
  chat_memory.py
  requirements.txt
  .gitignore

```

## Setup and Installation

### 1. Prerequisites

* Python 3.8 or newer
* `pip` (Python package installer)

### 2. Installation Steps

1.  **Clone or Download:**
    Download and extract the project files into a directory (e.g., `chatbot_project`).

2.  **Create and Activate a Virtual Environment (Recommended):**
    Open a terminal in the project directory and run:

    * **On Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    * **On macOS/Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install Dependencies:**
    Install all required packages from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1.  Ensure your virtual environment is activated.
2.  Run the main `interface.py` script from your terminal:

    ```bash
    python interface.py
    ```

**Note on First Run:** The *first time* you run the script, `transformers` will download and cache the `microsoft/DialoGPT-medium` model (approximately 800MB). This may take a few minutes depending on your internet connection. All subsequent launches will be immediate.

## Sample Interaction

The following is a sample session demonstrating the chatbot's conversational memory and special commands.


