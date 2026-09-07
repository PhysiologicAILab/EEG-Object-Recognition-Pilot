"""Low-level visual feature analysis of the 24 formal visible/occluded image pairs."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats


PROJECT_ROOT = Path(os.environ.get("EEG_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
PROJECT = Path(os.environ.get("EEG_EXPERIMENT_ROOT", PROJECT_ROOT / "experiment")).resolve()
OBJECTS_JSON = PROJECT / "data" / "objects.json"
RUN_TEMPLATE_DIR = PROJECT / "data" / "run_templates" / "v3_formal_six_objects_70occ"
OUT = Path(
    os.environ.get(
        "EEG_STIMULUS_FEATURE_OUTPUT",
        PROJECT_ROOT / "processed_data" / "stimulus_analysis" / "low_level_visual_features",
    )
).resolve()
PROTOCOL = "v3_formal_six_objects_70occ"
OBJECTS = ["bowl", "cup", "plate", "knife", "fork", "spoon"]

MANIFEST_ROOT = Path(
    os.environ.get(
        "EEG_RAW_BEHAVIOURAL_ROOT",
        PROJECT_ROOT / "raw_data" / "behavioural_audio",
    )
).resolve()
ACTUAL_MANIFESTS = sorted(MANIFEST_ROOT.rglob("trial_manifest.csv"))

EDGE_GAUSSIAN_KERNEL = (5, 5)
EDGE_GAUSSIAN_SIGMA = 1.0
EDGE_CANNY_LOW = 100
EDGE_CANNY_HIGH = 200
EDGE_SOBEL_APERTURE = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalise_relative_path(value: str) -> str:
    return str(value).replace("\\", "/").strip()


def load_formal_pairs() -> list[dict]:
    objects = json.loads(OBJECTS_JSON.read_text(encoding="utf-8"))
    if [item["id"] for item in objects] != OBJECTS:
        raise RuntimeError("objects.json does not contain the expected six formal categories in protocol order")
    pairs = []
    for item in objects:
        if len(item["exemplars"]) != 4:
            raise RuntimeError(f"{item['id']}: expected four formal exemplars")
        for exemplar in item["exemplars"]:
            if len(exemplar["occluded_paths"]) != 1:
                raise RuntimeError(f"{exemplar['id']}: expected exactly one formal occluded counterpart")
            visible_rel = normalise_relative_path(exemplar["visible_path"])
            occluded_rel = normalise_relative_path(exemplar["occluded_paths"][0])
            pairs.append({
                "instance_id": exemplar["id"], "object_category": item["id"],
                "visible_relative_path": visible_rel, "occluded_relative_path": occluded_rel,
                "visible_path": PROJECT / Path(visible_rel),
                "occluded_path": PROJECT / Path(occluded_rel),
            })
    if len(pairs) != 24 or len({x["instance_id"] for x in pairs}) != 24:
        raise RuntimeError("Formal pair definition is not exactly 24 unique instances")
    return pairs


def manifest_image_paths(path: Path) -> tuple[set[str], set[str]]:
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"protocol_version", "image_path", "is_practice", "exemplar_id", "visibility"}
    if not required.issubset(table.columns):
        raise RuntimeError(f"Missing required columns in {path}")
    formal = table[table.is_practice.eq("0")].copy()
    protocols = set(formal.protocol_version)
    if protocols != {PROTOCOL}:
        raise RuntimeError(f"Unexpected protocol in {path}: {protocols}")
    return set(formal.image_path.map(normalise_relative_path)), set(formal.exemplar_id)


def validate_identity(pairs: list[dict]) -> dict:
    expected_paths = {x["visible_relative_path"] for x in pairs} | {x["occluded_relative_path"] for x in pairs}
    expected_instances = {x["instance_id"] for x in pairs}
    if len(expected_paths) != 48:
        raise RuntimeError("objects.json does not define 48 unique formal image paths")
    for pair in pairs:
        for path in [pair["visible_path"], pair["occluded_path"]]:
            if not path.is_file():
                raise RuntimeError(f"Referenced formal image is missing: {path}")

    folder_visible = {normalise_relative_path(str(p.relative_to(PROJECT)))
                      for p in (PROJECT / "stimuli_final_candidates" / "clear").rglob("*.png")}
    folder_occluded = {normalise_relative_path(str(p.relative_to(PROJECT)))
                       for p in (PROJECT / "stimuli_final_candidates" / "occluded_70").rglob("*.png")}
    if folder_visible != {x["visible_relative_path"] for x in pairs}:
        raise RuntimeError("Visible folder contents differ from objects.json formal paths")
    if folder_occluded != {x["occluded_relative_path"] for x in pairs}:
        raise RuntimeError("Occluded folder contents differ from objects.json formal paths")

    template_paths, template_instances = set(), set()
    templates = sorted(RUN_TEMPLATE_DIR.glob("list_*.csv"))
    if len(templates) != 6:
        raise RuntimeError("Expected six frozen formal run-template files")
    for path in templates:
        paths, instances = manifest_image_paths(path)
        template_paths |= paths
        template_instances |= instances
    if template_paths != expected_paths or template_instances != expected_instances:
        raise RuntimeError("Frozen run templates do not use exactly the 24 formal pairs from objects.json")

    actual_paths, actual_instances = set(), set()
    for path in ACTUAL_MANIFESTS:
        if not path.is_file():
            raise RuntimeError(f"Actual formal trial manifest missing: {path}")
        paths, instances = manifest_image_paths(path)
        actual_paths |= paths
        actual_instances |= instances
    if actual_paths != expected_paths or actual_instances != expected_instances:
        raise RuntimeError("Actual participant manifests do not use exactly the 24 formal pairs")
    return {
        "objects_json_pairs": len(pairs), "unique_expected_paths": len(expected_paths),
        "frozen_run_templates": len(templates), "actual_manifest_files": len(ACTUAL_MANIFESTS),
        "template_paths_match": True, "actual_manifest_paths_match": True,
    }


def srgb_to_relative_luminance(rgb: np.ndarray) -> np.ndarray:
    srgb = rgb.astype(np.float64) / 255.0
    linear = np.where(srgb <= 0.04045, srgb / 12.92,
                      ((srgb + 0.055) / 1.055) ** 2.4)
    return 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]


def calculate_features(path: Path) -> dict:
    with Image.open(path) as image:
        if image.mode != "RGB" or image.size != (768, 768):
            raise RuntimeError(f"Unexpected image mode/size for {path}: {image.mode}, {image.size}")
        rgb = np.asarray(image, dtype=np.uint8)
    luminance = srgb_to_relative_luminance(rgb)
    mean_luminance = float(np.mean(luminance))
    rms_contrast = float(np.sqrt(np.mean(((luminance - mean_luminance) / mean_luminance) ** 2)))
    luminance_8bit = np.rint(np.clip(luminance, 0.0, 1.0) * 255.0).astype(np.uint8)
    blurred = cv2.GaussianBlur(
        luminance_8bit, EDGE_GAUSSIAN_KERNEL, EDGE_GAUSSIAN_SIGMA,
        borderType=cv2.BORDER_REFLECT_101,
    )
    edges = cv2.Canny(
        blurred, EDGE_CANNY_LOW, EDGE_CANNY_HIGH,
        apertureSize=EDGE_SOBEL_APERTURE, L2gradient=True,
    )
    edge_density = float(np.count_nonzero(edges) / edges.size)
    return {
        "width_px": rgb.shape[1], "height_px": rgb.shape[0],
        "mean_relative_luminance": mean_luminance,
        "rms_contrast": rms_contrast, "edge_density": edge_density,
    }


def rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def fmt_p(value: float) -> str:
    return "< .001" if value < .001 else f"= {value:.3f}"


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Non-empty output directory exists; refusing to overwrite: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    pairs = load_formal_pairs()
    identity_audit = validate_identity(pairs)
    paths = [path for pair in pairs for path in [pair["visible_path"], pair["occluded_path"]]]
    hashes_before = {str(path): sha256(path) for path in paths}

    rows = []
    for pair in pairs:
        visible_id = pair["visible_path"].stem
        occluded_id = pair["occluded_path"].stem
        for stimulus_type, path, relative_path, paired_id in [
            ("Visible", pair["visible_path"], pair["visible_relative_path"], occluded_id),
            ("Occluded", pair["occluded_path"], pair["occluded_relative_path"], visible_id),
        ]:
            rows.append({
                "image_id": path.stem, "instance_id": pair["instance_id"],
                "object_category": pair["object_category"], "stimulus_type": stimulus_type,
                "paired_image_id": paired_id, "relative_image_path": relative_path,
                **calculate_features(path), "image_sha256": hashes_before[str(path)],
            })
    images = pd.DataFrame(rows).sort_values(["object_category", "instance_id", "stimulus_type"],
                                            ascending=[True, True, False]).reset_index(drop=True)
    if len(images) != 48 or images.groupby("instance_id").size().ne(2).any():
        raise RuntimeError("Image-level output is not 24 complete pairs")

    metrics = ["mean_relative_luminance", "rms_contrast", "edge_density"]
    summary_rows = []
    for metric in metrics:
        for stimulus_type in ["Visible", "Occluded"]:
            values = images.loc[images.stimulus_type.eq(stimulus_type), metric].to_numpy(float)
            summary_rows.append({
                "metric": metric, "stimulus_type": stimulus_type, "N_images": len(values),
                "mean": values.mean(), "SD": values.std(ddof=1), "minimum": values.min(),
                "maximum": values.max(), "range": values.max() - values.min(),
            })
    group_summary = pd.DataFrame(summary_rows)

    comparison_rows = []
    wide = images.pivot(index="instance_id", columns="stimulus_type", values=metrics)
    for metric in metrics:
        visible = wide[(metric, "Visible")].to_numpy(float)
        occluded = wide[(metric, "Occluded")].to_numpy(float)
        difference = occluded - visible
        shapiro = stats.shapiro(difference)
        ttest = stats.ttest_rel(occluded, visible)
        dz = float(difference.mean() / difference.std(ddof=1))
        nonnormal = bool(shapiro.pvalue < 0.05)
        wilcoxon_stat = wilcoxon_p = rank_effect = np.nan
        if nonnormal:
            wilcoxon = stats.wilcoxon(difference, zero_method="wilcox", alternative="two-sided", method="auto")
            wilcoxon_stat = float(wilcoxon.statistic)
            wilcoxon_p = float(wilcoxon.pvalue)
            rank_effect = rank_biserial(difference)
        comparison_rows.append({
            "metric": metric, "N_pairs": len(difference),
            "visible_mean": visible.mean(), "occluded_mean": occluded.mean(),
            "mean_difference_occluded_minus_visible": difference.mean(),
            "SD_difference": difference.std(ddof=1),
            "paired_t": float(ttest.statistic), "paired_t_df": len(difference) - 1,
            "paired_t_p_two_sided": float(ttest.pvalue), "cohens_dz": dz,
            "shapiro_W_paired_difference": float(shapiro.statistic),
            "shapiro_p_paired_difference": float(shapiro.pvalue),
            "clear_nonnormality_flag_shapiro_p_lt_0_05": nonnormal,
            "wilcoxon_W_if_nonnormal": wilcoxon_stat,
            "wilcoxon_p_two_sided_if_nonnormal": wilcoxon_p,
            "matched_pairs_rank_biserial_if_nonnormal": rank_effect,
            "recommended_main_text_test": "Wilcoxon signed-rank" if nonnormal else "paired t-test",
        })
    comparisons = pd.DataFrame(comparison_rows)

    hashes_after = {str(path): sha256(path) for path in paths}
    if hashes_before != hashes_after:
        raise RuntimeError("One or more original image files changed during analysis")

    images.to_csv(OUT / "image_features.csv", index=False, encoding="utf-8-sig")
    group_summary.to_csv(OUT / "group_descriptives.csv", index=False, encoding="utf-8-sig")
    comparisons.to_csv(OUT / "paired_comparisons.csv", index=False, encoding="utf-8-sig")

    summary_by_metric = {row.metric: row for _, row in comparisons.iterrows()}
    method_lines = []
    for metric, row in summary_by_metric.items():
        if row.clear_nonnormality_flag_shapiro_p_lt_0_05:
            method_lines.append(
                f"- {metric}: paired differences failed Shapiro-Wilk, W = {row.shapiro_W_paired_difference:.3f}, "
                f"p {fmt_p(row.shapiro_p_paired_difference)}. Wilcoxon is recommended for the main text "
                f"(W = {row.wilcoxon_W_if_nonnormal:.1f}, p {fmt_p(row.wilcoxon_p_two_sided_if_nonnormal)}, "
                f"rank-biserial r = {row.matched_pairs_rank_biserial_if_nonnormal:.3f}); paired t is retained for transparency "
                f"(t(23) = {row.paired_t:.3f}, p {fmt_p(row.paired_t_p_two_sided)}, dz = {row.cohens_dz:.3f}).")
        else:
            method_lines.append(
                f"- {metric}: no clear Shapiro-Wilk departure, W = {row.shapiro_W_paired_difference:.3f}, "
                f"p {fmt_p(row.shapiro_p_paired_difference)}; paired t-test recommended, "
                f"t(23) = {row.paired_t:.3f}, p {fmt_p(row.paired_t_p_two_sided)}, dz = {row.cohens_dz:.3f}.")

    def group_value(metric, stimulus_type, column):
        return float(group_summary.loc[
            group_summary.metric.eq(metric) & group_summary.stimulus_type.eq(stimulus_type), column].iloc[0])

    paper_bits = []
    for metric, label in [("mean_relative_luminance", "mean relative luminance"),
                          ("rms_contrast", "RMS contrast"), ("edge_density", "edge density")]:
        row = summary_by_metric[metric]
        vis_mean, vis_sd = group_value(metric, "Visible", "mean"), group_value(metric, "Visible", "SD")
        occ_mean, occ_sd = group_value(metric, "Occluded", "mean"), group_value(metric, "Occluded", "SD")
        if row.recommended_main_text_test == "Wilcoxon signed-rank":
            test_text = (f"Wilcoxon W = {row.wilcoxon_W_if_nonnormal:.1f}, "
                         f"p {fmt_p(row.wilcoxon_p_two_sided_if_nonnormal)}, "
                         f"rank-biserial r = {row.matched_pairs_rank_biserial_if_nonnormal:.3f}")
        else:
            test_text = f"t(23) = {row.paired_t:.3f}, p {fmt_p(row.paired_t_p_two_sided)}, dz = {row.cohens_dz:.3f}"
        paper_bits.append(
            f"{label} (Visible: M = {vis_mean:.4f}, SD = {vis_sd:.4f}; "
            f"Occluded: M = {occ_mean:.4f}, SD = {occ_sd:.4f}; {test_text})")

    report = f"""# Low-level visual feature analysis of the formal stimuli

## Formal-image identity check

- Protocol: `{PROTOCOL}`.
- `objects.json` defined exactly **24 unique object instances**, each with one Visible and one 70%-Occluded image.
- Six frozen run templates and **{len(ACTUAL_MANIFESTS)} actual manifest files** jointly used exactly the same 48 paths.
- Categories: {', '.join(OBJECTS)}; four instances per category.
- Every PNG was RGB, 768 × 768 pixels, without an alpha channel.
- SHA-256 hashes for all 48 source images were unchanged before versus after analysis. No original image was modified.

## Feature definitions and implementation

All calculations used every pixel in the stored 768 × 768 stimulus image.

1. **Mean relative luminance.** Each 8-bit sRGB channel was scaled to [0,1] and linearised: `C_linear = C/12.92` when `C <= 0.04045`, otherwise `((C+0.055)/1.055)^2.4`. Relative luminance was `Y = 0.2126 R_linear + 0.7152 G_linear + 0.0722 B_linear` (range 0–1). Mean luminance was the arithmetic mean of Y across pixels.
2. **RMS contrast.** Dimensionless, mean-normalised RMS contrast was `sqrt(mean(((Y - mean(Y))/mean(Y))^2))`, equivalent to the population SD of Y divided by mean(Y).
3. **Edge density.** Y was mapped to 8-bit by rounding `255Y`, smoothed with OpenCV Gaussian blur (5 × 5 kernel, sigma = 1.0, `BORDER_REFLECT_101`), and processed by OpenCV Canny with fixed thresholds 100/200, Sobel aperture 3, and `L2gradient=True`. Edge density was `number of Canny edge pixels / total image pixels`. Thresholds were fixed before examining group results and were not tuned to obtain a desired outcome.

Software: Pillow {Image.__version__ if hasattr(Image, '__version__') else '10.4.0'}, NumPy {np.__version__}, OpenCV {cv2.__version__}, SciPy {__import__('scipy').__version__}.

## Group descriptives

The full mean, sample SD, minimum, maximum, and range for Visible and Occluded images are saved in `group_descriptives.csv`.

## Paired comparisons and normality check

Differences were defined as **Occluded minus Visible** for the same object instance. Cohen's dz was the mean paired difference divided by its sample SD. Shapiro-Wilk p < .05 was treated as a clear normality warning; where present, a two-sided Wilcoxon signed-rank test (`zero_method='wilcox'`) and matched-pairs rank-biserial correlation were also reported.

{chr(10).join(method_lines)}

No correction for the three feature-wise tests was added because this was a descriptive stimulus analysis; exact p values are reported and should not be treated as confirmatory hypothesis tests.

## Results summary

The 24 formally presented image pairs were compared on low-level visual properties. {paper_bits[0]}; {paper_bits[1]}; and {paper_bits[2]}. These comparisons describe the final stimulus set and do not establish that low-level features caused any EEG effect.

## Interpretive limitation

Although the visible and occluded stimuli were quantified for luminance, RMS contrast, and edge density, any residual low-level visual differences between paired images may contribute to condition-related neural differences and should therefore be considered when interpreting effects attributed to visual occlusion.

"""
    (OUT / "report.md").write_text(report, encoding="utf-8")
    parameters = {
        "protocol": PROTOCOL, "project_root": str(PROJECT), "identity_audit": identity_audit,
        "objects_json": str(OBJECTS_JSON), "run_template_directory": str(RUN_TEMPLATE_DIR),
        "actual_trial_manifests": [str(x) for x in ACTUAL_MANIFESTS],
        "image_count": 48, "pair_count": 24, "image_mode": "RGB", "image_size_px": [768, 768],
        "luminance": {
            "sRGB_linearisation": "C/12.92 if C<=0.04045 else ((C+0.055)/1.055)^2.4",
            "relative_luminance": "0.2126 R_linear + 0.7152 G_linear + 0.0722 B_linear",
            "mean": "arithmetic mean over all pixels",
        },
        "rms_contrast": "sqrt(mean(((Y-mean(Y))/mean(Y))^2)); dimensionless",
        "edge_density": {
            "input": "round(255*relative_luminance) uint8",
            "gaussian_kernel": list(EDGE_GAUSSIAN_KERNEL), "gaussian_sigma": EDGE_GAUSSIAN_SIGMA,
            "border": "OpenCV BORDER_REFLECT_101", "method": "OpenCV Canny",
            "low_threshold": EDGE_CANNY_LOW, "high_threshold": EDGE_CANNY_HIGH,
            "sobel_aperture": EDGE_SOBEL_APERTURE, "L2gradient": True,
            "density": "nonzero edge pixels / all image pixels",
        },
        "paired_difference": "Occluded minus Visible", "paired_test": "two-sided paired t-test",
        "effect_size": "Cohen's dz = mean paired difference / sample SD paired difference",
        "normality": "Shapiro-Wilk on paired differences; p<.05 flags clear departure",
        "nonnormal_fallback": "two-sided Wilcoxon signed-rank, zero_method=wilcox; matched-pairs rank-biserial",
        "source_sha256": hashes_before, "source_files_unchanged": hashes_before == hashes_after,
    }
    (OUT / "parameters.json").write_text(
        json.dumps(parameters, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "visual_features.py").write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({
        "output": str(OUT), "images": len(images), "pairs": images.instance_id.nunique(),
        "identity_audit": identity_audit,
        "comparisons": comparisons.to_dict(orient="records"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
