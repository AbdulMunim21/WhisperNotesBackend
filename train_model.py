from transformers import AutoModelForSeq2SeqLM, AutoConfig, AutoTokenizer, GenerationConfig
import os

# --- 1. Define Model and Path ---
model_name = "sshleifer/distilbart-cnn-12-6"
print(f"Selected model: {model_name}")

target_model_path = "./model"

# --- 2. Create Target Directory ---
print(f"\nEnsuring directory '{target_model_path}' exists...")
os.makedirs(target_model_path, exist_ok=True)
print(f"Directory ready.")

# --- 3. Download and Load Components ---
print(f"\nDownloading '{model_name}' from Hugging Face...")
try:
    config = AutoConfig.from_pretrained(model_name)
    print(f"Configuration loaded.")

    # CRITICAL: Load the tokenizer FIRST to get token IDs
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Tokenizer loaded.")
    print(f"  BOS Token ID from tokenizer: {tokenizer.bos_token_id}")
    print(f"  BOS Token: {tokenizer.bos_token}")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config)
    print("Model weights downloaded and loaded successfully.")

    # --- 4. Load or Create Generation Configuration with BOS Token Fix ---
    try:
        generation_config = GenerationConfig.from_pretrained(model_name)
        print("Found generation configuration on the hub.")
    except Exception as e:
        print(f"No specific generation config found on hub or error loading: {e}")
        print("Creating a default generation config suitable for summarization.")
        generation_config = GenerationConfig()

    # --- FIX: Ensure decoder_start_token_id is set in GenerationConfig ---
    # Priority: Use existing value, otherwise get from tokenizer, otherwise from model config
    if generation_config.decoder_start_token_id is None:
        if tokenizer.bos_token_id is not None:
            generation_config.decoder_start_token_id = tokenizer.bos_token_id
            print(f"  Set decoder_start_token_id from tokenizer.bos_token_id: {tokenizer.bos_token_id}")
        elif config.decoder_start_token_id is not None: # Fallback to model config
             generation_config.decoder_start_token_id = config.decoder_start_token_id
             print(f"  Set decoder_start_token_id from config.decoder_start_token_id: {config.decoder_start_token_id}")
        elif config.bos_token_id is not None: # Another fallback
             generation_config.decoder_start_token_id = config.bos_token_id
             print(f"  Set decoder_start_token_id from config.bos_token_id: {config.bos_token_id}")
        else:
            raise ValueError("Could not determine decoder_start_token_id from tokenizer or model config.")


    # Set other common summarization defaults if they aren't already set
    if generation_config.max_new_tokens is None:
         generation_config.max_new_tokens = 150
    if generation_config.min_new_tokens is None:
         generation_config.min_new_tokens = 30
    if generation_config.length_penalty is None:
         generation_config.length_penalty = 2.0
    if generation_config.num_beams is None:
         generation_config.num_beams = 4
    if generation_config.early_stopping is None:
         generation_config.early_stopping = True

    model.generation_config = generation_config
    print("Generation configuration resolved and associated with the model.")

except Exception as e:
    print(f"An error occurred during download or loading: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# --- 5. Save the Model and All Configurations Locally ---
print(f"\nSaving model and configurations to '{target_model_path}'...")
try:
    model.save_pretrained(target_model_path)
    print("  - Model weights saved.")

    config.save_pretrained(target_model_path)
    print("  - Model configuration saved.")

    tokenizer.save_pretrained(target_model_path) # Save tokenizer (includes bos_token_id info)
    print("  - Tokenizer saved.")

    # Save the FIXED generation configuration
    generation_config.save_pretrained(target_model_path)
    print("  - Generation configuration (WITH FIX) saved.")

    print("All components saved successfully.")

except Exception as e:
    print(f"An error occurred during saving: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# --- 6. Optional Verification (Load and Test Generate) ---
print("\n--- Verifying the saved model ---")
try:
    print("Reloading model and components from local storage...")
    reloaded_model = AutoModelForSeq2SeqLM.from_pretrained(target_model_path)
    reloaded_tokenizer = AutoTokenizer.from_pretrained(target_model_path)
    reloaded_gen_config = GenerationConfig.from_pretrained(target_model_path) # Load the saved one
    print("Reload successful.")

    print(f"\nChecking decoder_start_token_id in reloaded generation config:")
    print(f"  decoder_start_token_id: {getattr(reloaded_gen_config, 'decoder_start_token_id', 'NOT SET')}")
    print(f"  bos_token_id (from gen config): {getattr(reloaded_gen_config, 'bos_token_id', 'NOT SET')}")
    print(f"  bos_token_id (from model config): {getattr(reloaded_model.config, 'bos_token_id', 'NOT SET')}")
    print(f"  bos_token_id (from tokenizer): {reloaded_tokenizer.bos_token_id}")

    test_article = """The Amazon rainforest, also known as the lungs of the Earth, plays a crucial role in regulating the global climate.
It produces about 20% of the world's oxygen and absorbs large amounts of carbon dioxide.
However, deforestation poses a significant threat to this vital ecosystem.
Large areas are cleared for agriculture, logging, and mining, leading to biodiversity loss and increased greenhouse gas emissions.
Conservation efforts are essential to protect the Amazon and ensure its continued function in maintaining planetary health."""

    print("\nRunning a test summarization...")
    test_inputs = reloaded_tokenizer(test_article, return_tensors="pt", max_length=1024, truncation=True)

    # Use the model's explicitly set generation_config
    test_outputs = reloaded_model.generate(**test_inputs)
    generated_summary = reloaded_tokenizer.decode(test_outputs[0], skip_special_tokens=True)

    print(f"Generated Summary: {generated_summary}")
    print("Test summarization completed successfully.")

except Exception as e:
    print(f"An error occurred during verification: {e}")
    import traceback
    traceback.print_exc()

print(f"\n--- Process Complete ---")
print(f"Model is stored locally in: '{os.path.abspath(target_model_path)}'")
print("The generation configuration now includes the decoder_start_token_id fix.")