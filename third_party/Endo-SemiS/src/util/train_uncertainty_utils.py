import numpy as np
import torch
import torch.nn.functional as F


def enable_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout) or isinstance(module, torch.nn.Dropout2d):
            module.train()  # Keep dropout active


def estimate_uncertainty(model, batch, T=5):
    model.eval()  # Set model to evaluation mode (BN layers behave correctly)
    enable_dropout(model)  # Manually enable dropout layers

    predictions = []

    for _ in range(T):
        with torch.no_grad():
            logits = model(batch)  # Stochastic forward pass
            pred = torch.sigmoid(logits)  # Convert logits to probability
        predictions.append(pred)
    predictions = torch.stack(predictions)  # Shape: (T, batch_size, 1, H, W)
    mean_prediction = torch.mean(predictions, dim=0)  # Shape: (batch_size, 1, H, W)
    var_uncertainty = torch.var(predictions, dim=0)  # Shape: (batch_size, 1, H, W)
    eps = 1e-6  # Small value for numerical stability
    entropy_per_pass = - (predictions * torch.log(predictions + eps) + (1 - predictions) * torch.log(1 - predictions + eps))
    expected_entropy_uncertainty = torch.mean(entropy_per_pass, dim=0)  # Mean entropy over T runs

    return mean_prediction, var_uncertainty, expected_entropy_uncertainty



def get_pseudo_labels(model1, model2, input_labeled, input_unlabeled_w1, input_unlabeled_w2):
    with torch.no_grad():
        model1.eval()
        model2.eval()

        pseudo_outputs1 = (torch.sigmoid(model1(input_labeled)).detach() > 0.5).float()
        pseudo_outputs2 = (torch.sigmoid(model2(input_labeled)).detach() > 0.5).float()

        probability_w1_model1 = torch.sigmoid(model1(input_unlabeled_w1))
        pseudo_label_w1_model1 = (probability_w1_model1.detach() > 0.5).float()

        probability_w1_model2 = torch.sigmoid(model2(input_unlabeled_w1))
        pseudo_label_w1_model2 = (probability_w1_model2.detach() > 0.5).float()

        probability_w2_model1 = torch.sigmoid(model1(input_unlabeled_w2))
        pseudo_label_w2_model1 = (probability_w2_model1.detach() > 0.5).float()

        probability_w2_model2 = torch.sigmoid(model2(input_unlabeled_w2))
        pseudo_label_w2_model2 = (probability_w2_model2.detach() > 0.5).float()

        pseudo_labels_labeled = {'model1': pseudo_outputs1, 'model2': pseudo_outputs2}
        pseudo_labels_unlabeled_w1 = {'model1': pseudo_label_w1_model1, 'model2': pseudo_label_w1_model2}
        pseudo_labels_unlabeled_w2 = {'model1': pseudo_label_w2_model1, 'model2': pseudo_label_w2_model2}

        probability_unlabeled_w1 = {'model1': probability_w1_model1, 'model2': probability_w1_model2}
        probability_unlabeled_w2 = {'model1': probability_w2_model1, 'model2': probability_w2_model2}

    return pseudo_labels_labeled, pseudo_labels_unlabeled_w1, pseudo_labels_unlabeled_w2, probability_unlabeled_w1, probability_unlabeled_w2


def get_threshold(data_tensor, cutting_point=0.95):
    mean_val = data_tensor.mean(dim=tuple(range(1, data_tensor.ndim)), keepdim=True)
    std_val = data_tensor.std(dim=tuple(range(1, data_tensor.ndim)), keepdim=True)
    raw_threshold = mean_val + std_val
    cap_threshold = torch.quantile(
        data_tensor.view(data_tensor.shape[0], -1), cutting_point, dim=1, keepdim=True
    ).view(*([data_tensor.shape[0]] + [1] * (data_tensor.ndim - 1)))

    threshold = torch.min(raw_threshold, cap_threshold)
    return threshold

def get_threshold_batch(data_tensor, cutting_point=0.95):
    raw_threshold = data_tensor.mean()
    cap_threshold = torch.quantile(data_tensor.view(-1), cutting_point)
    threshold = torch.min(raw_threshold, cap_threshold)
    return threshold

def masked_bce_with_logits(input, target, mask, eps=1e-8):
    """
    input: logits tensor, shape [B, 1, H, W] or [B, C, H, W]
    target: pseudo-label (0/1) tensor, same shape
    mask: 0/1 or continuous weighting mask, same shape
    Returns: scalar, the average BCE loss over the *nonzero* mask pixels.
    """
    bce_per_pixel = F.binary_cross_entropy_with_logits(
        input, target, reduction='none'
    )
    weighted_sum = (bce_per_pixel * mask).sum()
    normalizer = mask.sum() + eps
    return weighted_sum / normalizer

def sharpening(P, T=0.1):
    return P ** (1 / T) / (P ** (1 / T) + (1 - P) ** (1 / T))


def entropy_binary_minmax(prob, eps=1e-8):
    # Compute entropy
    entropy = - (prob * torch.log(prob + eps) + (1 - prob) * torch.log(1 - prob + eps))

    # Min-Max normalize per image
    min_val = torch.amin(entropy, dim=(1, 2, 3), keepdim=True)
    max_val = torch.amax(entropy, dim=(1, 2, 3), keepdim=True)
    normalized_entropy = (entropy - min_val) / (max_val - min_val + eps)

    return normalized_entropy


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(args, epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)


def split_tensor_into_patches(tensor, patch_size=128):
    """
    Splits an image tensor into non-overlapping patches.

    Args:
        tensor (torch.Tensor): Input tensor of shape [B, C, H, W].
        patch_size (int): Size of each square patch.

    Returns:
        patches (torch.Tensor): Shape [B * num_patches, C, patch_size, patch_size].
    """
    B, C, H, W = tensor.shape

    # Ensure image is divisible by patch size
    assert H % patch_size == 0 and W % patch_size == 0, \
        f"Image dimensions ({H}, {W}) must be divisible by patch size ({patch_size})"

    # Compute number of patches along height and width
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size

    # Unfold (extract patches efficiently)
    patches = tensor.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)

    # Reshape to [B, num_patches_h, num_patches_w, C, patch_size, patch_size]
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()

    # Reshape to [B * num_patches, C, patch_size, patch_size]
    patches = patches.view(-1, C, patch_size, patch_size)

    return patches


def reconstruct_from_patches(patches, patch_size=128, batch_size=1):
    """
    Reconstructs an image from patches.

    Args:
        patches (torch.Tensor): Shape [B * num_patches, C, patch_size, patch_size].
        patch_size (int): Size of each patch.

    Returns:
        reconstructed (torch.Tensor): Shape [B, C, H, W] (original size before splitting).
    """
    Bp, C, H, W = patches.shape  # Bp = B * num_patches
    num_patches_per_image = Bp // batch_size  # Total patches per image

    # Compute number of patches along height and width
    num_patches_h = int((num_patches_per_image) ** 0.5)
    num_patches_w = num_patches_h  # Assuming square patches

    # Compute original batch size
    #B = Bp // (num_patches_h * num_patches_w)
    B = batch_size

    # Reshape back to [B, num_patches_h, num_patches_w, C, patch_size, patch_size]
    patches = patches.view(B, num_patches_h, num_patches_w, C, patch_size, patch_size)

    # Permute to [B, C, num_patches_h, patch_size, num_patches_w, patch_size]
    patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()

    # Reshape to reconstruct original image [B, C, H, W]
    reconstructed = patches.view(B, C, num_patches_h * patch_size, num_patches_w * patch_size)

    return reconstructed


