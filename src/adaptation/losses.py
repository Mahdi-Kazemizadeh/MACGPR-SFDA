import torch


class AdaptationLosses:
    """Loss functions used in source-free domain adaptation."""

    @staticmethod
    def entropy_loss(probabilities: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """Compute mean prediction entropy.

        Lower entropy encourages confident predictions.
        """
        entropy = -torch.sum(
            probabilities * torch.log(probabilities + eps),
            dim=1,
        )
        return entropy.mean()

    @staticmethod
    def diversity_loss(probabilities: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """Compute diversity loss.

        This term discourages class collapse by encouraging the mean prediction
        distribution to remain spread across classes.
        """
        mean_prediction = probabilities.mean(dim=0)
        return torch.sum(mean_prediction * torch.log(mean_prediction + eps))
