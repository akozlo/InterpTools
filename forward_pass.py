def list_activation_tensors(model, prompt):
    import re
    import torch

    # Perform a forward pass with the input prompt
    tokens = model.to_tokens(prompt)
    logits, cache = model.run_with_cache(tokens)

    # Collect unique activation names and their shapes after replacing block numbers
    activations = {}
    for key, value in cache.items():
        # Replace block numbers with 'n' using regular expressions
        new_key = re.sub(r'^blocks\.\d+', 'blocks.n', key)
        if isinstance(value, torch.Tensor):
            # Get shape without the first dimension (batch size)
            shape = value.shape[1:]
        elif isinstance(value, list):
            shape = value[0].shape[1:] if value else 'N/A'
        else:
            shape = 'Unknown'

        # Store the shape in the activations dictionary
        # Only store the key if it hasn't been stored before
        if new_key not in activations:
            activations[new_key] = shape

    # Determine the maximum length of activation names
    max_length = max(len(name) for name in activations.keys())

    # Display available activation tensors with aligned columns
    print("Available activation tensors:")
    for key, shape in activations.items():
        print(f"{key:<{max_length}}   {shape}")



def list_weight_matrices(model):
    import re
    import torch

    # Collect unique keys and their shapes after replacing block numbers
    state_dict = model.state_dict()
    keys_shapes = {}
    seen_keys = set()

    for key, value in state_dict.items():
        # Replace block numbers with 'n' using regular expressions
        new_key = re.sub(r'^blocks\.\d+', 'blocks.n', key)

        # Store the shape of the tensor
        shape = value.shape

        # Only add the key if it's not already seen to avoid duplicates
        if new_key not in seen_keys:
            seen_keys.add(new_key)
            keys_shapes[new_key] = shape

    # Determine the maximum length of keys for alignment
    max_length = max(len(k) for k in keys_shapes.keys())

    # Display the keys and their shapes with aligned columns
    print("Model state_dict keys:")
    for key in keys_shapes.keys():
        shape = keys_shapes[key]
        print(f"{key:<{max_length}}   {shape}")
