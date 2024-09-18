from transformer_lens import HookedTransformer
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from IPython.display import display, HTML
import io
import base64
import random

def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def generate_text_with_probs(model, prompt, max_new_tokens, temperature=1.0, random_seed=None):
    set_seed(random_seed)
    input_tokens = model.to_tokens(prompt, prepend_bos=True)
    generated_tokens = input_tokens.clone()
    token_probs = []

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(generated_tokens)[:, -1, :] / temperature
            next_token_probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, num_samples=1)

        generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
        token_probs.append(next_token_probs[0, next_token.item()].item())

        if next_token.item() == model.tokenizer.eos_token_id:
            break

    generated_text = model.to_string(generated_tokens[0])
    clean_tokens = model.to_str_tokens(generated_tokens[0])
    
    # Remove the initial prompt tokens from clean_tokens and token_probs
    prompt_length = len(model.to_str_tokens(input_tokens[0]))
    clean_tokens = clean_tokens[prompt_length:]
    token_probs = token_probs[:len(clean_tokens)]
    
    return generated_text, clean_tokens, token_probs


def map_prob_to_color(prob, min_prob=0.0, max_prob=1.0, alpha=0.6):
    """
    Maps a probability to a color between red and green with transparency.
    
    Args:
        prob (float): Probability between 0 and 1.
        min_prob (float): Minimum probability for scaling.
        max_prob (float): Maximum probability for scaling.
        alpha (float): Transparency level (0 to 1, where 1 is opaque).
    
    Returns:
        str: RGBA color code.
    """
    # Clamp the probability between min_prob and max_prob
    prob = max(min(prob, max_prob), min_prob)
    # Normalize the probability between 0 and 1
    normalized = (prob - min_prob) / (max_prob - min_prob)
    
    # Calculate red and green components
    red = int((1 - normalized) * 255)
    green = int(normalized * 255)
    blue = 0  # Keeping blue constant for a gradient between red and green

    return f'rgba({red}, {green}, {blue}, {alpha})'

def create_html_visualization(generated_text, token_probs, model):
    tokens = model.to_str_tokens(generated_text)
    
    if len(tokens) != len(token_probs):
        min_len = min(len(tokens), len(token_probs))
        tokens = tokens[:min_len]
        token_probs = token_probs[:min_len]
    
    html = """
    <div style="font-family: monospace; white-space: pre-wrap;">
        <div style="margin-bottom: 5px;">
    """
    
    for prob in token_probs:
        prob_percent = f"{prob * 100:.1f}%"
        html += f'<span style="display: inline-block; width: 60px; text-align: center; font-size: 10px; color: #555;">{prob_percent}</span>'
    html += "</div><div>"

    for token, prob in zip(tokens, token_probs):
        color = map_prob_to_color(prob)
        display_token = token if token.strip() != '' else '&nbsp;'
        html += f'<span style="background-color: {color}; padding: 2px 4px; margin: 1px; border-radius: 3px;">{display_token}</span>'

    html += "</div></div>"
    return html

def display_colored_text(generated_text, clean_tokens, token_probs):
    html = """
    <div style="font-family: monospace; white-space: pre-wrap;">
        <div style="margin-bottom: 5px;">
    """
    
    for token, prob in zip(clean_tokens, token_probs):
        prob_percent = f"{prob * 100:.1f}%"
        color = map_prob_to_color(prob)
        
        display_token = token if token else '&nbsp;'
        
        html += f'<span style="display: inline-block; margin: 2px; padding: 2px 4px; background-color: {color}; border-radius: 3px;">'
        html += f'<span style="font-size: 10px; color: #555;">{prob_percent}</span><br>'
        html += f'{display_token}</span>'

    html += "</div></div>"
    display(HTML(html))

def get_top_n_tokens(model, prompt, n, temperature=1.0, random_seed=None):
    set_seed(random_seed)
    input_tokens = model.to_tokens(prompt)
    
    with torch.no_grad():
        logits = model(input_tokens)[:, -1, :] / temperature
        probs = torch.nn.functional.softmax(logits, dim=-1)
        top_n = torch.topk(probs, n)
    
    results = [(prompt, None)]
    for i in range(n):
        token = model.to_string(top_n.indices[0][i])
        prob = top_n.values[0][i].item()
        results.append((token, prob))
    
    return results

def generate_text(model, prompt, max_length, temperature=1.0, random_seed=None):
    set_seed(random_seed)
    input_tokens = model.to_tokens(prompt)
    
    output_tokens = model.generate(
        input_tokens,
        max_new_tokens=max_length - len(input_tokens),
        temperature=temperature,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )
    
    return model.to_string(output_tokens[0])

def clean_token(token):
    """Clean up a token for display."""
    return token.replace('Ġ', ' ').replace(' ', ' ').strip() or '[SPACE]'

def top_n_viz(results):
    """
    Create a visualization of the top n tokens and their probabilities,
    displaying a Matplotlib bar chart and a styled Pandas table side-by-side.
    
    Args:
        results (list of tuples): Output from get_top_n_tokens function, containing (token, probability) pairs
    
    Returns:
        None (displays the plot and table side-by-side)
    """
    prompt = results[0][0]
    print(f"{prompt} ____________")
    tokens, probs = zip(*results[1:])
    num_tokens = len(tokens)
    
    clean_tokens = [clean_token(token) for token in tokens]
    
    buf = io.BytesIO()
    
    plt.figure(figsize=(8, 4))
    bars = plt.bar(range(num_tokens), probs, color='cornflowerblue')
    
    plt.ylabel("P(token)", fontsize=14)
    
    plt.xticks(range(num_tokens), clean_tokens, rotation=45, ha='right', fontsize=12)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.3f}',
            ha='center',
            va='bottom',
            fontsize=12
        )
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    img_html = f'<img src="data:image/png;base64,{image_base64}" width="600"/>'
    
    df = pd.DataFrame([(clean_token(token), f"{prob:.3f}") for token, prob in results[1:]], columns=["Token", "Probability"])
    
    styled_table = df.style \
        .set_table_styles([
            {'selector': 'th', 
             'props': [('font-size', '14px'), 
                       ('text-align', 'center'), 
                       ('border-bottom', '2px solid #d3d3d3')]},
            {'selector': 'td', 
             'props': [('font-size', '14px'), 
                       ('text-align', 'center'), 
                       ('border-bottom', '1px solid #d3d3d3')]}
        ]) \
        .set_properties(**{
            'border-collapse': 'collapse',
            'border': '1px solid #d3d3d3'
        })
    
    table_html = styled_table.to_html(index=False)
    
    table_html = table_html.replace(r'\.0+<', '<')
    
    combined_html = f"""
    <div style="display: flex; align-items: flex-start;">
        <div style="margin-right: 50px;">
            {img_html}
        </div>
        <div>
            {table_html}
        </div>
    </div>
    """
    
    display(HTML(combined_html))


def get_token_probabilities_per_layer(
    model: HookedTransformer,
    input_text: str,
    target_word: str,
    random_seed=None
) -> dict:
    """
    Computes the probability of a target word at each layer of a HookedTransformer model.

    Args:
        model (HookedTransformer): The HookedTransformer model to use.
        input_text (str): The input text leading up to the target word.
        target_word (str): The word whose probability is to be computed.

    Returns:
        dict: A dictionary mapping layer numbers to the probability of the target word.
    """
    set_seed(random_seed)
    # Tokenize input and target
    input_tokens = model.to_tokens(input_text, prepend_bos=True)
    target_token = model.to_single_token(target_word)

    # Get the number of layers
    num_layers = model.cfg.n_layers

    # Initialize dictionary to store probabilities
    probabilities = {}

    # Define a hook function to capture intermediate outputs
    def hook_fn(activation, hook):
        layer_num = int(hook.name.split('.')[1])
        residual = activation
        normalized = model.blocks[layer_num].ln2(residual)
        logits = model.unembed(model.ln_final(normalized))
        probs = torch.softmax(logits, dim=-1)
        probabilities[layer_num] = probs[0, -1, target_token].item()

    # Create a list of hook points
    hook_points = [f"blocks.{i}.hook_resid_post" for i in range(num_layers)]

    # Run the model with hooks
    with torch.no_grad():
        _, cache = model.run_with_cache(
            input_tokens,
            return_type="logits",
            names_filter=hook_points
        )

    # Apply the hook function to each layer's output
    for name, activation in cache.items():
        if name in hook_points:
            hook_fn(activation, SimpleNamespace(name=name))

    return probabilities

class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def layer_probs_viz(probabilities):
    """
    Create two side-by-side line graphs showing probabilities and log probabilities by layer.

    Args:
        probabilities (dict): A dictionary mapping layer numbers to probabilities.

    Returns:
        None (displays the plot)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    layers = list(probabilities.keys())
    probs = list(probabilities.values())
    logprobs = np.log(probs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4.5))

    # Probability plot
    ax1.plot(layers, probs, marker='o')
    ax1.set_title('Probability by Layer')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Probability')
    ax1.grid(True)

    # Log probability plot
    ax2.plot(layers, logprobs, marker='o', color='orange')
    ax2.set_title('Log Probability by Layer')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Log Probability')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def top_probs_by_layer(model, input_text, temperature=1.0, random_seed=None):
    """
    Compute and visualize the most probable next-token and its probability for each layer of the model.

    Args:
        model (HookedTransformer): The HookedTransformer model.
        input_text (str): The input text leading up to the target word.
        temperature (float, optional): Temperature for softmax. Defaults to 1.0.

    Returns:
        pd.DataFrame: A DataFrame containing the results.
    """
    set_seed(random_seed)
    # Tokenize input
    input_tokens = model.to_tokens(input_text, prepend_bos=True)

    # Get the number of layers
    num_layers = model.cfg.n_layers

    # Initialize lists to store results
    top_tokens = []
    top_probabilities = []

    # Define a hook function to capture intermediate outputs
    def hook_fn(activation, hook):
        layer_num = int(hook.name.split('.')[1])
        residual = activation
        normalized = model.blocks[layer_num].ln2(residual)
        logits = model.unembed(normalized)
        probs = torch.softmax(logits / temperature, dim=-1)  # Apply temperature to softmax
        top_n = torch.topk(probs[0, -1], 1)  # Focus on the last token
        top_tokens.append(model.to_string(top_n.indices[0]))
        top_probabilities.append(top_n.values[0].item())

    # Create a list of hook points
    hook_points = [f"blocks.{i}.hook_resid_post" for i in range(num_layers)]

    # Run the model with hooks
    with torch.no_grad():
        _, cache = model.run_with_cache(
            input_tokens,               
            return_type="logits",
            names_filter=hook_points
        )

    # Apply the hook function to each layer's output
    for name, activation in cache.items():
        if name in hook_points:
            hook_fn(activation, SimpleNamespace(name=name))

    # Create a DataFrame to store results
    layers = list(range(1, num_layers + 1))  # Layer numbers from 1 to num_layers
    results = pd.DataFrame({
        'Layer': layers,
        'Top Token': top_tokens,
        'Top Probability': top_probabilities
    })  

    # Create a styled table
    styled_table = results.style \
        .format({'Top Probability': '{:.3f}'}) \
        .set_table_styles([
            {'selector': 'th', 'props': [('font-size', '12.5px'), ('text-align', 'center'), ('border-bottom', '2px solid #d3d3d3')]},
            {'selector': 'td', 'props': [('font-size', '12.5px'), ('text-align', 'center'), ('border-bottom', '1px solid #d3d3d3')]}
        ]) \
        .set_properties(**{'border-collapse': 'collapse', 'border': '1px solid #d3d3d3'})

    # Display the styled table
    display(styled_table)
    
    return results

class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)