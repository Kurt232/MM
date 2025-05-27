import os
import json
import glob
import sys

from pytablewriter import MarkdownTableWriter, CsvTableWriter
from collections import defaultdict

def make_results_table(model_result_dict):
    """Generate both Markdown and CSV tables of results grouped by task, version, and metric."""
    # Common headers and data preparation
    headers = ["Task", "Version", "Metric", "Model", "Value", "", "Stderr"]
    values = []

    # Grouping logic (same as before)
    grouped_data = defaultdict(lambda: defaultdict(list))
    for model_name, result_dict in model_result_dict.items():
        results = result_dict.get("results", {})
        versions = result_dict.get("versions", {})
        for task in sorted(results.keys()):
            task_data = results[task]
            version = versions.get(task, "")
            key = (task, version)
            for metric in task_data:
                if metric.endswith("_stderr"):
                    continue
                value = task_data[metric]
                stderr = task_data.get(f"{metric}_stderr", None)
                grouped_data[key][metric].append((model_name, value, stderr))

    # Row construction (same as before)
    for (task, version) in sorted(grouped_data.keys()):
        metrics_dict = grouped_data[(task, version)]
        first_metric = True
        for metric in sorted(metrics_dict.keys()):
            models_values = metrics_dict[metric]
            models_values.sort(key=lambda x: x[0])
            for i, (model, val, se) in enumerate(models_values):
                current_task = task if (first_metric and i == 0) else ""
                current_version = version if (first_metric and i == 0) else ""
                current_metric = metric if i == 0 else ""
                
                row = [
                    current_task,
                    current_version,
                    current_metric,
                    model,
                    f"{val:.4f}",
                    "±" if se is not None else "",
                    f"{se:.4f}" if se is not None else ""
                ]
                values.append(row)
                if first_metric and i == 0:
                    first_metric = False

    # Generate both formats
    md_writer = MarkdownTableWriter()
    md_writer.headers = headers
    md_writer.value_matrix = values

    csv_writer = CsvTableWriter()
    csv_writer.headers = headers
    csv_writer.value_matrix = values

    return md_writer.dumps(), csv_writer.dumps()

if __name__ == "__main__":
    assert len(sys.argv) >= 2 and len(sys.argv) <= 3
    root_path = sys.argv[1]
    is_markdown = True
    if len(sys.argv) == 3:
        is_markdown = False
    root_path = os.path.join(root_path, "results")
    model_results_dirs = os.listdir(root_path)
    model_results_dirs = [d for d in model_results_dirs]
    data_dict = {}
    for d in model_results_dirs:
        path = f"{root_path}/{d}/*"
        # results_2025-05-25T01-27-56.047822.json
        timestamps = [p.split('/')[-1][8:-5] for p in glob.glob(path)]
        timestamp = sorted(timestamps)[-1]
        data = json.load(open(f"{root_path}/{d}/results_{timestamp}.json"))
        model_name = d.split("_")[-1]
        data_dict[model_name] = data
    
    for model_name, data in data_dict.items():
        data["results"].pop("all", None)
        for task in list(data["results"].keys()):
            if task.startswith("helm|mmlu:") and not task.startswith("helm|mmlu:_"):
                data["results"].pop(task)
    
    mk_table, csv_table = make_results_table(data_dict)
    if is_markdown:
        print(mk_table)
    else:
        print(csv_table)