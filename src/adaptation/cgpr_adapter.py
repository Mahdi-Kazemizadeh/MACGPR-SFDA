import gc
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score
from torch.utils.data import DataLoader, Dataset, Subset

from src.adaptation.query_selection import QuerySelector
from src.adaptation.reliability import ReliabilityEstimator
from src.adaptation.losses import AdaptationLosses
from src.utils.io import ExperimentIO
from src.utils.metrics import ClassificationMetrics


class LabelOverrideDataset(Dataset):
    """Wraps a subset and replaces its labels with refined pseudo-labels."""

    def __init__(self, subset: Subset, labels: np.ndarray) -> None:
        self.subset = subset
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int):
        image, _ = self.subset[index]
        return image, int(self.labels[index])


class CGPRAdapter:
    """Cluster-Guided Pseudo-label Refinement for source-free adaptation."""

    def __init__(
        self,
        model: torch.nn.Module,
        target_dataset: Dataset,
        config: dict[str, Any],
        device: torch.device,
    ) -> None:
        self.model = model
        self.target_dataset = target_dataset
        self.config = config
        self.device = device

        self.dataset_config = config["dataset"]
        self.loader_config = config["loader"]
        self.adaptation_config = config["adaptation"]
        self.output_config = config["output"]

        self.num_classes = int(self.dataset_config["num_classes"])

        self.history: dict[str, list[Any]] = {
            "accuracy": [],
            "pseudo_error": [],
            "silhouette": [],
            "coverage": [],
            "label_changes": [],
            "entropy": [],
            "threshold": [],
            "learning_rate": [],
            "iteration_time_seconds": [],
            "selection_score": [],
            "mean_reliability": [],
            "reliability_threshold": [],
            "easy_count": [],
            "hard_count": [],
            "unsafe_count": [],
            "hard_ratio": [],
        }
        self.reliability_estimator = ReliabilityEstimator(
            num_classes=self.num_classes,
            config=self.config,
        )
        self.query_selector = QuerySelector(config=self.config)

    def adapt(self) -> dict[str, Any]:
        self.model.to(self.device)

        self.model.freeze_classifier()
        self.model.unfreeze_feature_extractor()

        optimizer = torch.optim.Adam(
            filter(lambda parameter: parameter.requires_grad,
                   self.model.parameters()),
            lr=float(self.adaptation_config["lr_backbone"]),
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(self.adaptation_config["iterations"]),
            eta_min=1e-6,
        )

        base_loader = self._build_base_loader()

        print("Computing source-only target accuracy...")
        logits_np, features_np, true_labels = self._full_pass(base_loader)
        initial_predictions = logits_np.argmax(axis=1)
        initial_accuracy = float(accuracy_score(
            true_labels, initial_predictions))
        print(f"Source-only target accuracy: {initial_accuracy:.4f}")

        best_unsupervised_score = None
        best_state = deepcopy(self.model.state_dict())
        best_debug_accuracy = initial_accuracy

        threshold = float(self.adaptation_config["initial_threshold"])
        total_start_time = time.perf_counter()

        for iteration in range(1, int(self.adaptation_config["iterations"]) + 1):
            iteration_start_time = time.perf_counter()

            if iteration > 1:
                logits_np, features_np, _ = self._full_pass(base_loader)

            probabilities = F.softmax(
                torch.from_numpy(logits_np), dim=1).numpy()
            confidence = probabilities.max(axis=1)
            pseudo_labels = probabilities.argmax(axis=1)

            entropy = self._entropy_numpy(probabilities)
            average_entropy = float(entropy.mean())

            normalized_all_features = self._l2_normalize(features_np)
            all_clusters = self._cluster_features(normalized_all_features)
            cluster_refined_all_labels = self._refine_labels_by_cluster(
                pseudo_labels=pseudo_labels,
                clusters=all_clusters,
            )

            reliability_scores = self.reliability_estimator.compute(
                probabilities=probabilities,
                features=features_np,
                pseudo_labels=pseudo_labels,
                clusters=all_clusters,
                cluster_refined_labels=cluster_refined_all_labels,
            )

            query_selection = self.query_selector.select(reliability_scores)

            reliability_threshold = float(
                self.adaptation_config.get("reliability", {}).get(
                    "reliability_threshold",
                    threshold,
                )
            )

            confident_mask = query_selection.easy_mask.copy()

            min_confident_samples = int(
                self.adaptation_config["min_confident_samples"]
            )

            if confident_mask.sum() < min_confident_samples:
                reliability_threshold = max(0.50, reliability_threshold - 0.05)
                confident_mask = reliability_scores > reliability_threshold

            coverage = float(confident_mask.mean())
            mean_reliability = float(
                reliability_scores[confident_mask].mean()) if confident_mask.sum() > 0 else 0.0

            self.history["coverage"].append(coverage)
            self.history["entropy"].append(average_entropy)
            self.history["threshold"].append(round(threshold, 4))
            self.history["reliability_threshold"].append(
                round(reliability_threshold, 4))
            self.history["mean_reliability"].append(mean_reliability)
            self.history["learning_rate"].append(
                optimizer.param_groups[0]["lr"])
            self.history["easy_count"].append(query_selection.easy_count)
            self.history["hard_count"].append(query_selection.hard_count)
            self.history["unsafe_count"].append(query_selection.unsafe_count)
            self.history["hard_ratio"].append(
                query_selection.hard_count / len(reliability_scores)
            )

            if confident_mask.sum() == 0:
                current_accuracy = self._evaluate_accuracy(
                    base_loader, true_labels)
                self._record_empty_iteration(
                    current_accuracy=current_accuracy,
                    iteration_start_time=iteration_start_time,
                )
                scheduler.step()
                continue

            selected_indices = np.where(confident_mask)[0]
            selected_features = features_np[confident_mask]
            selected_pseudo_labels = pseudo_labels[confident_mask]
            selected_true_labels = true_labels[confident_mask]

            normalized_features = normalized_all_features[confident_mask]

            clusters = all_clusters[confident_mask]
            refined_labels = cluster_refined_all_labels[confident_mask]

            label_changes = int(
                (refined_labels != selected_pseudo_labels).sum())
            pseudo_error = 1.0 - float(
                accuracy_score(selected_true_labels, refined_labels)
            )
            silhouette = self._safe_silhouette(normalized_features, clusters)

            self.history["label_changes"].append(label_changes)
            self.history["pseudo_error"].append(pseudo_error)
            self.history["silhouette"].append(silhouette)

            self._train_on_refined_labels(
                selected_indices=selected_indices,
                refined_labels=refined_labels,
                optimizer=optimizer,
            )

            scheduler.step()

            current_accuracy = self._evaluate_accuracy(
                base_loader, true_labels)
            self.history["accuracy"].append(current_accuracy)

            selection_score = self._compute_unsupervised_selection_score(
                coverage=coverage,
                average_entropy=average_entropy,
                label_changes=label_changes,
                selected_count=len(selected_indices),
                silhouette=silhouette,
            )

            self.history["selection_score"].append(selection_score)

            iteration_time = time.perf_counter() - iteration_start_time
            self.history["iteration_time_seconds"].append(
                round(iteration_time, 3))

            status = ""
            if best_unsupervised_score is None or selection_score > best_unsupervised_score:
                best_unsupervised_score = selection_score
                best_state = deepcopy(self.model.state_dict())
                status = "best-unsup"

            best_debug_accuracy = max(best_debug_accuracy, current_accuracy)
            print(
                f"Iter {iteration}/{self.adaptation_config['iterations']} | "
                f"thr={threshold:.3f} | "
                f"cov={coverage:.3f} | "
                f"rel={mean_reliability:.3f} | "
                f"easy={query_selection.easy_count} | "
                f"hard={query_selection.hard_count} | "
                f"changes={label_changes} | "
                f"pseudo_err={pseudo_error:.4f} | "
                f"sil={silhouette if silhouette is not None else 'NA'} | "
                f"acc={current_accuracy:.4f} | "
                f"time={ExperimentIO.format_seconds(iteration_time)} "
                f"{status}"
            )

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        total_time = time.perf_counter() - total_start_time

        self.model.load_state_dict(best_state)

        final_logits, _, final_true_labels = self._full_pass(base_loader)
        final_predictions = final_logits.argmax(axis=1)

        final_metrics = ClassificationMetrics.compute(
            y_true=final_true_labels,
            y_pred=final_predictions,
            num_classes=self.num_classes,
        )

        final_metrics["best_debug_accuracy_during_adaptation"] = best_debug_accuracy
        final_metrics["initial_source_only_accuracy"] = initial_accuracy
        final_metrics["best_unsupervised_selection_score"] = best_unsupervised_score
        final_metrics["selection_is_sfda_clean"] = True
        final_metrics["history"] = self.history
        final_metrics["total_time_seconds"] = round(total_time, 3)
        final_metrics["class_names"] = self.dataset_config["class_names"]

        self._save_results(final_metrics, final_predictions, final_true_labels)

        return final_metrics

    def _build_base_loader(self) -> DataLoader:
        return DataLoader(
            self.target_dataset,
            batch_size=int(self.loader_config["batch_size"]),
            shuffle=False,
            num_workers=int(self.loader_config["num_workers"]),
            pin_memory=bool(self.loader_config["pin_memory"]),
        )

    @torch.no_grad()
    def _full_pass(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()

        logits_list = []
        features_list = []
        labels_list = []

        for images, labels in loader:
            images = images.to(self.device)

            logits, features = self.model(images, return_features=True)

            logits_list.append(logits.cpu().numpy().astype(np.float32))
            features_list.append(features.cpu().numpy().astype(np.float32))
            labels_list.append(labels.numpy())

        return (
            np.concatenate(logits_list, axis=0),
            np.concatenate(features_list, axis=0),
            np.concatenate(labels_list, axis=0),
        )

    def _cluster_features(self, features: np.ndarray) -> np.ndarray:
        n_samples = features.shape[0]
        k_clusters = int(self.adaptation_config["k_clusters"])
        k_clusters = max(2, min(k_clusters, n_samples - 1))

        kmeans = KMeans(
            n_clusters=k_clusters,
            n_init=10,
            random_state=int(self.config["seed"]),
        )

        return kmeans.fit_predict(features)

    def _refine_labels_by_cluster(
        self,
        pseudo_labels: np.ndarray,
        clusters: np.ndarray,
    ) -> np.ndarray:
        refined_labels = pseudo_labels.copy()

        for cluster_id in np.unique(clusters):
            cluster_mask = clusters == cluster_id
            majority_label = np.bincount(
                pseudo_labels[cluster_mask],
                minlength=self.num_classes,
            ).argmax()
            refined_labels[cluster_mask] = majority_label

        return refined_labels

    def _train_on_refined_labels(
        self,
        selected_indices: np.ndarray,
        refined_labels: np.ndarray,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        subset = Subset(self.target_dataset, selected_indices.tolist())
        train_dataset = LabelOverrideDataset(subset, refined_labels)

        train_loader = DataLoader(
            train_dataset,
            batch_size=int(self.loader_config["batch_size"]),
            shuffle=True,
            num_workers=int(self.loader_config["num_workers"]),
            pin_memory=bool(self.loader_config["pin_memory"]),
        )

        self.model.train()

        for _ in range(int(self.adaptation_config["train_epochs_per_iteration"])):
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits, _ = self.model(images, return_features=True)
                probabilities = F.softmax(logits, dim=1)

                loss_ce = F.cross_entropy(logits, labels)
                loss_entropy = AdaptationLosses.entropy_loss(probabilities)
                loss_diversity = AdaptationLosses.diversity_loss(probabilities)

                loss = (
                    loss_ce
                    + float(self.adaptation_config["entropy_weight"]) * loss_entropy
                    + float(self.adaptation_config["diversity_weight"]) * loss_diversity
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    @torch.no_grad()
    def _evaluate_accuracy(
        self,
        loader: DataLoader,
        true_labels: np.ndarray,
    ) -> float:
        self.model.eval()

        predictions = []

        for images, _ in loader:
            images = images.to(self.device)
            logits = self.model(images)
            predictions.append(logits.argmax(dim=1).cpu().numpy())

        predictions_np = np.concatenate(predictions, axis=0)
        return float(accuracy_score(true_labels, predictions_np))

    def _record_empty_iteration(
        self,
        current_accuracy: float,
        iteration_start_time: float,
    ) -> None:
        self.history["accuracy"].append(current_accuracy)
        self.history["pseudo_error"].append(None)
        self.history["silhouette"].append(None)
        self.history["label_changes"].append(0)

        iteration_time = time.perf_counter() - iteration_start_time
        self.history["iteration_time_seconds"].append(round(iteration_time, 3))

    def _save_results(
        self,
        metrics: dict[str, Any],
        predictions: np.ndarray,
        true_labels: np.ndarray,
    ) -> None:
        results_dir = ExperimentIO.ensure_dir(
            self.output_config["results_dir"])

        ExperimentIO.save_json(metrics, results_dir / "metrics.json")
        ExperimentIO.save_json(self.history, results_dir / "history.json")

        np.save(results_dir / "predictions.npy", predictions)
        np.save(results_dir / "true_labels.npy", true_labels)

        torch.save(self.model.state_dict(), results_dir / "model_state.pth")

    def _compute_unsupervised_selection_score(
        self,
        coverage: float,
        average_entropy: float,
        label_changes: int,
        selected_count: int,
        silhouette: float | None,
    ) -> float:
        """Compute a target-label-free score for model selection.

        This score is used only for unsupervised checkpoint selection.
        It does not use target labels.
        """
        normalized_entropy = average_entropy / np.log(self.num_classes)

        if selected_count <= 0:
            change_penalty = 1.0
        else:
            change_penalty = label_changes / selected_count

        silhouette_score_value = 0.0 if silhouette is None else float(
            silhouette)

        score = (
            coverage
            + silhouette_score_value
            - normalized_entropy
            - change_penalty
        )

        return float(score)

    @staticmethod
    def _l2_normalize(features: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        return features / (np.linalg.norm(features, axis=1, keepdims=True) + eps)

    @staticmethod
    def _entropy_numpy(probabilities: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        return -np.sum(probabilities * np.log(probabilities + eps), axis=1)

    @staticmethod
    def _safe_silhouette(
        features: np.ndarray,
        clusters: np.ndarray,
        max_samples: int = 5000,
    ) -> float | None:
        if len(np.unique(clusters)) <= 1:
            return None

        if features.shape[0] <= 1:
            return None

        if features.shape[0] > max_samples:
            indices = np.random.choice(
                features.shape[0],
                size=max_samples,
                replace=False,
            )
            features = features[indices]
            clusters = clusters[indices]

        return float(silhouette_score(features, clusters))
