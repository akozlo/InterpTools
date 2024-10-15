import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display_html
from transformer_lens import HookedTransformer



def activation_agg_sim(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """
    Computes the cosine similarity between two activation tensors after averaging over the sequence dimension.

    Args:
        tensor1 (torch.Tensor): Activation tensor of shape (batch_size, seq_length1, hidden_size)
        tensor2 (torch.Tensor): Activation tensor of shape (batch_size, seq_length2, hidden_size)

    Returns:
        torch.Tensor: Cosine similarity tensor of shape (batch_size,)
    """
    # Ensure tensors are on the same device
    if tensor1.device != tensor2.device:
        tensor2 = tensor2.to(tensor1.device)
    
    # Average over the sequence dimension (dim=1)
    avg_tensor1 = tensor1.mean(dim=1)  # Shape: (batch_size, hidden_size)
    avg_tensor2 = tensor2.mean(dim=1)  # Shape: (batch_size, hidden_size)

    # Compute cosine similarity along the hidden_size dimension
    cos_sim = torch.nn.functional.cosine_similarity(avg_tensor1, avg_tensor2, dim=-1)  # Shape: (batch_size,)

    return cos_sim


def compare_activation_similarity(model, prompt1, prompt2, mlp=True, attention=True, resid=True, layernorm=True):
    """
    Compare activation patterns between two prompts in a transformer model.

    Args:
        model (HookedTransformer): The transformer model.
        prompt1 (str): The first prompt.
        prompt2 (str): The second prompt.
        mlp (bool): Whether to include MLP activations. Default is True.
        attention (bool): Whether to include attention activations. Default is True.
        resid (bool): Whether to include residual stream activations. Default is True.

    Returns:
        message (str): Note about prompt lengths and calculation method.
        layer_similarities (dict): Dictionary of per-layer aggregate similarity scores.
        layer_pairwise_similarities (dict): Dictionary of per-layer pairwise similarity scores.
        cumulative_similarity (float): Cumulative aggregate similarity score across all layers.
        cumulative_pairwise_similarity (float): Cumulative pairwise similarity score across all layers.
        final_token_similarities (dict): Dictionary of per-layer final token similarity scores.
        cumulative_final_token_similarity (float): Cumulative final token similarity across all layers.
    """

    # Tokenize the prompts
    tokens1 = model.to_tokens(prompt1)  # Shape: (1, seq_len1)
    tokens2 = model.to_tokens(prompt2)  # Shape: (1, seq_len2)

    seq_len1 = tokens1.shape[1]
    seq_len2 = tokens2.shape[1]

    # Determine if prompts have equal length
    if seq_len1 == seq_len2:
        message = ("Note: prompt1 and prompt2 are equal length. "
                   "Aggregate similarity and flattened sequence similarity may still differ "
                   "due to differences in how positional activations are compared.")
    else:
        message = ("Note: prompt1 and prompt2 have different token lengths. "
                   "Activation similarity is therefore calculated through aggregating or padding sequences.")

    # Build the list of activation names we want to analyze
    activation_names = []
    
    # Add non-layer-specific activations
    activation_names.extend(['hook_embed', 'hook_pos_embed'])
    
    for i in range(model.cfg.n_layers):
        # Add layer-specific activations
        activation_names.extend([
            f'blocks.{i}.hook_resid_pre',
            f'blocks.{i}.ln1.hook_scale',
            f'blocks.{i}.ln1.hook_normalized',
            f'blocks.{i}.attn.hook_q',
            f'blocks.{i}.attn.hook_k',
            f'blocks.{i}.attn.hook_v',
            f'blocks.{i}.attn.hook_attn_scores',
            f'blocks.{i}.attn.hook_pattern',
            f'blocks.{i}.attn.hook_z',
            f'blocks.{i}.hook_attn_out',
            f'blocks.{i}.hook_resid_mid',
            f'blocks.{i}.ln2.hook_scale',
            f'blocks.{i}.ln2.hook_normalized',
            f'blocks.{i}.mlp.hook_pre',
            f'blocks.{i}.mlp.hook_post',
            f'blocks.{i}.hook_mlp_out',
            f'blocks.{i}.hook_resid_post'
        ])
    
    # Add final layer norm activations
    activation_names.extend(['ln_final.hook_scale', 'ln_final.hook_normalized'])

    # Filter activations based on user preferences
    if not mlp:
        activation_names = [name for name in activation_names if 'mlp' not in name]
    if not attention:
        activation_names = [name for name in activation_names if 'attn' not in name]
    if not resid:
        activation_names = [name for name in activation_names if 'resid' not in name]
    if not layernorm:
        activation_names = [name for name in activation_names if 'ln' not in name]

    # Run the model with caching for both prompts
    _, cache1 = model.run_with_cache(tokens1)
    _, cache2 = model.run_with_cache(tokens2)

    # Extract the desired activations from the caches
    activations1 = {name: cache1[name] for name in activation_names}
    activations2 = {name: cache2[name] for name in activation_names}

    # Initialize variables to store similarity scores
    layer_similarities = {}
    cumulative_similarity = 0.0

    layer_padded_similarities = {}
    cumulative_padded_similarity = 0.0

    final_token_similarities = {}
    cumulative_final_token_similarity = 0.0

    total_layers = len(activation_names)

    for name in activation_names:
        # Activations have shape (1, seq_len, hidden_size)
        act1 = activations1[name]  # Shape: (1, seq_len1, hidden_size)
        act2 = activations2[name]  # Shape: (1, seq_len2, hidden_size)

        ### Aggregate Similarity ###
        # Compute mean over the sequence dimension to get shape (1, hidden_size)
        act1_mean = act1.mean(dim=1)  # Shape: (1, hidden_size)
        act2_mean = act2.mean(dim=1)  # Shape: (1, hidden_size)

        # Remove the batch dimension
        act1_mean = act1_mean.squeeze(0)  # Shape: (hidden_size,)
        act2_mean = act2_mean.squeeze(0)

        # Ensure activations are on the same device
        act1_mean = act1_mean.to(act2_mean.device)

        # Compute cosine similarity between the two mean vectors
        aggregate_similarity = F.cosine_similarity(act1_mean, act2_mean, dim=0).item()
        layer_similarities[name] = aggregate_similarity
        cumulative_similarity += aggregate_similarity

        ### Pairwise Cosine Similarity ###
        pairwise_similarities = []
        for token1 in act1:
            for token2 in act2:
                similarity = F.cosine_similarity(token1, token2, dim=0).item()
                pairwise_similarities.append(similarity)
        
        # Calculate the mean of all pairwise similarities
        pairwise_similarity = sum(pairwise_similarities) / len(pairwise_similarities)
        
        # Store the pairwise similarity for this layer
        layer_pairwise_similarities[name] = pairwise_similarity
        cumulative_pairwise_similarity += pairwise_similarity

        ### Final Token Similarity ###
        # Get the final token activations
        final_act1 = act1[:, -1, :]  # Shape: (1, hidden_size)
        final_act2 = act2[:, -1, :]  # Shape: (1, hidden_size)

        # Remove batch dimension
        final_act1 = final_act1.squeeze(0)  # Shape: (hidden_size,)
        final_act2 = final_act2.squeeze(0)

        # Ensure activations are on the same device
        final_act1 = final_act1.to(final_act2.device)

        # Compute cosine similarity
        final_similarity = F.cosine_similarity(final_act1, final_act2, dim=0).item()
        final_token_similarities[name] = final_similarity
        cumulative_final_token_similarity += final_similarity

    # Compute average cumulative similarities
    cumulative_similarity /= total_layers
    cumulative_padded_similarity /= total_layers
    cumulative_final_token_similarity /= total_layers

    return (
        message,
        layer_similarities,
        layer_pairwise_similarities,
        cumulative_similarity,
        cumulative_pairwise_similarity,
        final_token_similarities,
        cumulative_final_token_similarity
    )


def display_activation_similarity_tables(model, prompt1, prompt2, mlp=True, attention=True, resid=True):
    """
    Computes activation similarities between two prompts and displays the results
    in pandas DataFrames, organized by activation type and similarity metric.

    Args:
        model (HookedTransformer): The transformer model.
        prompt1 (str): The first prompt.
        prompt2 (str): The second prompt.
        mlp (bool): Whether to include MLP activations. Default is True.
        attention (bool): Whether to include attention activations. Default is True.
        resid (bool): Whether to include residual stream activations. Default is True.

    Returns:
        None
    """
    # Invoke the comparison function with the specified activation types
    results = compare_activation_similarity(model, prompt1, prompt2, mlp=mlp, attention=attention, resid=resid)

    # Unpack the results
    (message,
     layer_similarities,
     layer_pairwise_similarities,
     cumulative_similarity,
     cumulative_pairwise_similarity,
     final_token_similarities,
     cumulative_final_token_similarity) = results

    # Display the message
    print(message)

    # Determine which activation types are included
    activation_types = []
    if resid:
        activation_types.append('hook_resid_post')
    if attention:
        activation_types.append('hook_attn_out')
    if mlp:
        activation_types.append('hook_mlp_out')

    # Organize similarities by activation type
    aggregate_similarities_by_type = {}
    padded_similarities_by_type = {}
    final_token_similarities_by_type = {}

    for activation_type in activation_types:
        # Extract similarities for this activation type
        aggregate_similarities_by_type[activation_type] = {
            layer: value for layer, value in layer_similarities.items() if activation_type in layer
        }
        padded_similarities_by_type[activation_type] = {
            layer: value for layer, value in layer_padded_similarities.items() if activation_type in layer
        }
        final_token_similarities_by_type[activation_type] = {
            layer: value for layer, value in final_token_similarities.items() if activation_type in layer
        }

    # Compute cumulative similarities per activation type
    cumulative_similarities_by_type = {}
    cumulative_padded_similarities_by_type = {}
    cumulative_final_token_similarities_by_type = {}

    for activation_type in activation_types:
        agg_sims = aggregate_similarities_by_type[activation_type].values()
        cumulative_similarities_by_type[activation_type] = sum(agg_sims) / len(agg_sims) if agg_sims else 0.0

        pad_sims = padded_similarities_by_type[activation_type].values()
        cumulative_padded_similarities_by_type[activation_type] = sum(pad_sims) / len(pad_sims) if pad_sims else 0.0

        final_sims = final_token_similarities_by_type[activation_type].values()
        cumulative_final_token_similarities_by_type[activation_type] = sum(final_sims) / len(final_sims) if final_sims else 0.0

    # Set display options for better readability
    pd.set_option('display.precision', 4)

    # Function to display DataFrames side by side
    def display_side_by_side(dfs, captions):
        html_str = ''
        for df, caption in zip(dfs, captions):
            html_str += f'<div style="display:inline-block; margin-right:20px; font-size: 9.5pt;">'
            html_str += f'<h3>{caption}</h3>'
            html_str += df.to_html(index=False)
            html_str += '</div>'
        display_html(html_str, raw=True)

    # For each activation type, create dataframes and display them
    for activation_type in activation_types:
        # Create dataframes
        df_aggregate = pd.DataFrame.from_dict(aggregate_similarities_by_type[activation_type], orient='index', columns=['Aggregate Similarity'])
        df_padded = pd.DataFrame.from_dict(padded_similarities_by_type[activation_type], orient='index', columns=['Padded Similarity'])
        df_final_token = pd.DataFrame.from_dict(final_token_similarities_by_type[activation_type], orient='index', columns=['Final Token Similarity'])

        # Add cumulative similarities
        df_aggregate.loc['Cumulative'] = cumulative_similarities_by_type[activation_type]
        df_padded.loc['Cumulative'] = cumulative_padded_similarities_by_type[activation_type]
        df_final_token.loc['Cumulative'] = cumulative_final_token_similarities_by_type[activation_type]

        # Reset index and rename columns
        df_aggregate.reset_index(inplace=True)
        df_padded.reset_index(inplace=True)
        df_final_token.reset_index(inplace=True)

        df_aggregate.rename(columns={'index': 'Layer'}, inplace=True)
        df_padded.rename(columns={'index': 'Layer'}, inplace=True)
        df_final_token.rename(columns={'index': 'Layer'}, inplace=True)

        # Prepare the tables and captions
        tables = [df_aggregate, df_padded, df_final_token]
        captions = [
            f'Aggregate Similarity ({activation_type})',
            f'Padded Similarity ({activation_type})',
            f'Final Token Similarity ({activation_type})'
        ]

        # Display the tables
        display_side_by_side(tables, captions)



################################################################################


def display_activation_similarity_plots(model, prompt1, prompt2, mlp=True, attention=True, resid=True, logscale=False):
    """
    Computes activation similarities between two prompts and displays the results
    in line graphs organized by activation type, with cumulative scores represented as
    semi-transparent dashed lines.

    Args:
        model (HookedTransformer): The transformer model.
        prompt1 (str): The first prompt.
        prompt2 (str): The second prompt.
        mlp (bool): Whether to include MLP activations. Default is True.
        attention (bool): Whether to include attention activations. Default is True.
        resid (bool): Whether to include residual stream activations. Default is True.
        logscale (bool): If True, uses log(similarity) as the y-axis values. Default is False.

    Returns:
        None
    """
    # Invoke the comparison function with the specified activation types
    results = compare_activation_similarity(model, prompt1, prompt2, mlp=mlp, attention=attention, resid=resid)

    # Unpack the results
    (message,
     layer_similarities,
     layer_pairwise_similarities,
     cumulative_similarity,
     cumulative_pairwise_similarity,
     final_token_similarities,
     cumulative_final_token_similarity) = results

    # Display the message
    print(message)

    # Determine which activation types are included
    activation_types = []
    if resid:
        activation_types.append('hook_resid_post')
    if attention:
        activation_types.append('hook_attn_out')
    if mlp:
        activation_types.append('hook_mlp_out')

    # Organize similarities by activation type
    similarities_by_type = {}
    cumulative_similarities_by_type = {}

    for activation_type in activation_types:
        # Extract similarities for this activation type
        similarities_by_type[activation_type] = {
            'aggregate': {layer: value for layer, value in layer_similarities.items() if activation_type in layer},
            'padded': {layer: value for layer, value in layer_padded_similarities.items() if activation_type in layer},
            'final_token': {layer: value for layer, value in final_token_similarities.items() if activation_type in layer},
        }

        # Compute cumulative similarities
        agg_sims = similarities_by_type[activation_type]['aggregate'].values()
        cumulative_agg = sum(agg_sims) / len(agg_sims) if agg_sims else 0.0

        pad_sims = similarities_by_type[activation_type]['padded'].values()
        cumulative_padded = sum(pad_sims) / len(pad_sims) if pad_sims else 0.0

        final_sims = similarities_by_type[activation_type]['final_token'].values()
        cumulative_final = sum(final_sims) / len(final_sims) if final_sims else 0.0

        cumulative_similarities_by_type[activation_type] = {
            'aggregate': cumulative_agg,
            'padded': cumulative_padded,
            'final_token': cumulative_final
        }

    # Prepare data for plotting
    activation_labels = {
        'hook_resid_post': 'Residual Stream',
        'hook_attn_out': 'Attention Output',
        'hook_mlp_out': 'MLP Output'
    }

    max_num_ticks = 10  # Maximum number of x-axis ticks to display

    # Plot settings
    plot_types = ['aggregate', 'padded', 'final_token']
    plot_titles = {
        'aggregate': 'Aggregate Similarity',
        'padded': 'Padded Similarity',
        'final_token': 'Final Token Similarity'
    }
    plot_colors = {
        'aggregate': 'b',
        'padded': 'g',
        'final_token': 'm'
    }

    # Apply log transformation if logscale is True
    epsilon = 1e-10  # Small value to prevent log(0)

    # Create subplots
    num_activation_types = len(activation_types)
    fig, axs = plt.subplots(num_activation_types, 3, figsize=(18, 3.5 * num_activation_types))

    if num_activation_types == 1:
        axs = [axs]  # Ensure axs is a list when there's only one activation type

    for idx, activation_type in enumerate(activation_types):
        for i, plot_type in enumerate(plot_types):
            # Extract values
            similarities = similarities_by_type[activation_type][plot_type]
            layers = list(similarities.keys())
            layer_numbers = list(range(len(layers)))
            values = np.array(list(similarities.values()))
            cumulative_value = cumulative_similarities_by_type[activation_type][plot_type]

            # Apply log transformation if needed
            if logscale:
                values = np.log(values + epsilon)
                cumulative_value = np.log(cumulative_value + epsilon)
                y_label = 'log(Similarity)'
            else:
                y_label = 'Similarity'

            # Plot
            axs[idx][i].set_ylim(-1, 0)  # Set y-axis range from 0 to 1 when logscale=False
            axs[idx][i].plot(layer_numbers, values, marker='o', label=plot_titles[plot_type], color=plot_colors[plot_type])
            axs[idx][i].axhline(y=cumulative_value, color='r', linestyle='--', alpha=0.5,
                                label=f'Cumulative: {cumulative_value:.4f}')
            axs[idx][i].set_title(f"{plot_titles[plot_type]} ({activation_labels[activation_type]})")
            axs[idx][i].set_xlabel('Layer')
            axs[idx][i].set_ylabel(y_label)
            # Adjust x-axis ticks
            if len(layer_numbers) > max_num_ticks:
                step = max(1, len(layer_numbers) // max_num_ticks)
                axs[idx][i].set_xticks(layer_numbers[::step])
            else:
                axs[idx][i].set_xticks(layer_numbers)
            axs[idx][i].set_xticklabels([str(n) for n in axs[idx][i].get_xticks()])  # Convert x-ticks to strings

            if not logscale:
                axs[idx][i].set_ylim(0, 1)  # Set y-axis range from 0 to 1 when logscale=False
            # When logscale=True, y-axis is scaled automatically

            axs[idx][i].legend()
            axs[idx][i].grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()
