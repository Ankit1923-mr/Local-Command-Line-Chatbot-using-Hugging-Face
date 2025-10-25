import sys
from model_loader import load_model
from chat_memory import ChatMemory
from colorama import init, Fore, Style

init(autoreset=True)

def strip_bot_text(full_text, prompt_suffix="Bot:"):
    reply = full_text.split(prompt_suffix)[-1]
    
    
    # Remove trailing 'User:' or repeated 'Bot:'
    for stop_word in ["User:", "Bot:", "System:"]:
        if stop_word in reply:
            reply = reply.split(stop_word)[0]
    
    # Take first line only to avoid rambling
    reply = reply.strip().split("\n")[0]
    return reply.strip()

def main():
    print(Fore.GREEN + "Starting local CLI chatbot (type /exit to quit, /clear to reset memory)\n")
    
    # Set device to use CPU
    print(Fore.YELLOW + "Device set to use cpu\n")
    
    # Load model and tokenizer
    generator = load_model("microsoft/DialoGPT-medium", device=-1)
    
    # Initialize memory with sliding window of 5 turns
    memory = ChatMemory(max_turns=5)
    
    # DO NOT pre-seed memory - let it build naturally from conversation
    # This demonstrates dynamic memory management
    
    while True:
        try:
            # Get user input
            user_input = input(Fore.CYAN + "User: " + Style.RESET_ALL)
            
            # Handle special commands
            if user_input.strip().lower() == "/exit":
                print(Fore.GREEN + "Exiting chatbot. Goodbye!")
                break
            
            if user_input.strip().lower() == "/clear":
                memory.clear()
                print(Fore.YELLOW + "Memory cleared.")
                continue
            
            # Skip empty inputs
            if not user_input.strip():
                continue
            
            # Build prompt with conversation history
            context = memory.get_context()
            if context:
                prompt = f"{context}\nUser: {user_input}\nBot:"
            else:
                prompt = f"User: {user_input}\nBot:"
            
            # Generate response
            outputs = generator(
                prompt,
                max_new_tokens=50,
                do_sample=True,
                top_p=0.9,
                temperature=0.6,
                pad_token_id=generator.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            # Extract bot reply
            raw_reply = outputs[0]['generated_text']
            bot_reply = strip_bot_text(prompt + raw_reply, "Bot:")
            
            # Handle empty responses
            if not bot_reply:
                bot_reply = "I'm not sure how to respond to that."
            
            # Display bot response with memory indicator
            memory_info = f"[Memory: {len(memory.history)}/{memory.history.maxlen} turns]"
            print(Fore.MAGENTA + "Bot: " + Style.RESET_ALL + bot_reply)
            print(Fore.WHITE + Style.DIM + memory_info + Style.RESET_ALL)
            
            # Add to memory
            memory.add(user_input, bot_reply)
        
        except KeyboardInterrupt:
            print(Fore.GREEN + "\n\nExiting chatbot. Goodbye!")
            break
        except EOFError:
            print(Fore.GREEN + "\n\nExiting chatbot. Goodbye!")
            break
        except Exception as e:
            print(Fore.RED + f"Error: {str(e)}")
            print(Fore.YELLOW + "Continuing...")

if __name__ == "__main__":
    main()
