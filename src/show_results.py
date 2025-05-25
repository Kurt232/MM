import os
import json
import glob
import sys

from pytablewriter import MarkdownTableWriter
from collections import defaultdict

def make_results_table(model_result_dict):
    """Generate table of results grouped by task, version, and metric."""
    md_writer = MarkdownTableWriter()
    md_writer.headers = ["Task", "Version", "Metric", "Model", "Value", "", "Stderr"]

    # Group data by (task, version), then by metric
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

    values = []
    # Iterate through each (task, version) group
    for (task, version) in sorted(grouped_data.keys()):
        metrics_dict = grouped_data[(task, version)]
        first_metric = True
        for metric in sorted(metrics_dict.keys()):
            models_values = metrics_dict[metric]
            # Sort models alphabetically
            models_values.sort(key=lambda x: x[0])
            for i, (model, val, se) in enumerate(models_values):
                # Determine displayed task, version, and metric
                current_task = task if (first_metric and i == 0) else ""
                current_version = version if (first_metric and i == 0) else ""
                current_metric = metric if i == 0 else ""
                
                if se is not None:
                    row = [
                        current_task,
                        current_version,
                        current_metric,
                        model,
                        f"{val:.4f}",
                        "±",
                        f"{se:.4f}",
                    ]
                else:
                    row = [
                        current_task,
                        current_version,
                        current_metric,
                        model,
                        f"{val:.4f}",
                        "",
                        "",
                    ]
                values.append(row)
                # After the first row, subsequent metrics don't show task/version
                if first_metric and i == 0:
                    first_metric = False

    md_writer.value_matrix = values
    return md_writer.dumps()

if __name__ == "__main__":
    len(sys.argv) == 2 
    root_path = sys.argv[1]
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
    
    table = make_results_table(data_dict)
    print(table)