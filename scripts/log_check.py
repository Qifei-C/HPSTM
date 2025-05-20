import os
import re
import argparse
from collections import defaultdict

def parse_log_file(log_path, val_indicator_keywords, metric_preferences_config):
    all_metrics_values = defaultdict(list)

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line_lower = line.lower()
                has_val_indicator = any(val_kw.lower() in line_lower for val_kw in val_indicator_keywords)

                if has_val_indicator:
                    for match in re.finditer(r'\b([a-zA-Z0-9_./-]+)\b\s*[:=]\s*(-?\d+\.?\d*(?:e[-+]?\d+)?)\b', line_lower):
                        metric_key = match.group(1).strip()
                        try:
                            metric_value = float(match.group(2))
                            all_metrics_values[metric_key].append(metric_value)
                        except (ValueError, TypeError):
                            pass

                    for val_kw in val_indicator_keywords:
                        val_kw_l = val_kw.lower()
                        # Regex: val_keyword ( optionally_ আরো_words ) number
                        # The metric name part allows for multiple words separated by space/underscore/dot/hyphen
                        # We capture the val_keyword, the subsequent name parts, and the number.
                        # Example: val_kw_l = "val"
                        # line_lower = "... val some metric name 0.123 ..."
                        # group(1) from format: val_kw_l
                        # group(2): " some metric name" (parts after val_kw, before number)
                        # group(m.lastindex) or group(3) here: the number
                        pattern_str = r'\b({})\b([\s_.-]+[a-zA-Z0-9_.-]+(?:[\s_.-]+[a-zA-Z0-9_.-]+)*)?\s+(-?\d+\.?\d*(?:e[-+]?\d+)?)\b'
                        for m in re.finditer(pattern_str.format(re.escape(val_kw_l)), line_lower):
                            metric_name_part = ""
                            if m.group(2): 
                                metric_name_part = m.group(2).strip()
                                metric_name_part = "_".join(re.split(r'[\s_.-]+', metric_name_part))
                            
                            if metric_name_part:
                                metric_key = f"{val_kw_l}_{metric_name_part}"
                            else: # If no specific name part, could be val_kw itself implies a default metric
                                  # This case might be too ambiguous, but let's allow it.
                                  # Or, more likely, the pattern structure means group(2) is what makes the metric unique.
                                  # If metric_name_part is empty, perhaps this match is not specific enough.
                                  # For now, we require a metric_name_part.
                                continue # Skip if no distinct name part after val_kw

                            metric_key = metric_key.replace('__', '_').strip('_')

                            try:
                                metric_value = float(m.group(m.lastindex))
                                all_metrics_values[metric_key].append(metric_value)
                            except (ValueError, TypeError):
                                pass
    except FileNotFoundError:
        print(f"Error: {log_path} is missing.")
        return {}
    except Exception as e:
        print(f"Error: loading {log_path} with error: {e}")
        return {}

    best_metrics_found = {}
    for metric_key, values_list in all_metrics_values.items():
        if not values_list:
            continue

        is_higher_better = None
        if metric_key in metric_preferences_config:
            is_higher_better = metric_preferences_config[metric_key]
        else:
            sorted_pref_keys = sorted([k for k in metric_preferences_config if k != '__default__'], key=len, reverse=True)
            for pref_substr in sorted_pref_keys:
                if pref_substr in metric_key:
                    is_higher_better = metric_preferences_config[pref_substr]
                    break

        if is_higher_better is None:
            is_higher_better = metric_preferences_config.get('__default__', True)

        if is_higher_better:
            best_metrics_found[metric_key] = max(values_list)
        else:
            best_metrics_found[metric_key] = min(values_list)
            
    return best_metrics_found

def main():
    parser = argparse.ArgumentParser(description="Loads and parses .log files from subfolders, extracting the best values for all validation metrics in each log.")
    parser.add_argument("--log_dir", type=str, default=r".\Manifold\results\logs",
                        help="Base directory containing experiment subfolders and .log files.")
    parser.add_argument("--val_keywords", type=str, default="val,validation,eval,test",
                        help="Comma-separated keywords indicating validation/test lines (e.g., 'val,validation,eval,test').")
    parser.add_argument("--higher_better_metrics", type=str, default="acc,accuracy,f1,f1_score,f1-score,precision,recall,auc,map,score,dice,iou",
                        help="Comma-separated metric names (or substrings) where higher values are better.")
    parser.add_argument("--lower_better_metrics", type=str, default="loss,err,error",
                        help="Comma-separated metric names (or substrings) where lower values are better.")
    parser.add_argument("--default_metric_preference", type=str, choices=['high', 'low'], default='high',
                        help="Default preference ('high' or 'low') for metrics not explicitly specified in the lists above. Defaults to 'high'.")

    args = parser.parse_args()

    base_log_dir = args.log_dir
    val_indicator_keywords = [kw.strip() for kw in args.val_keywords.split(',') if kw.strip()]
    
    metric_preferences_config = {'__default__': args.default_metric_preference == 'high'}
    for m_str_group, is_high in [(args.lower_better_metrics, False), (args.higher_better_metrics, True)]:
        for m_str in m_str_group.split(','):
            clean_m_str = m_str.strip().lower()
            if clean_m_str:
                metric_preferences_config[clean_m_str] = is_high
    
    if not os.path.isdir(base_log_dir):
        print(f"Error: Log directory '{base_log_dir}' not found.")
        return

    all_results = []

    print(f"Scanning for log files in subfolders of: {base_log_dir}")
    print(f"Looking for all metrics in lines containing any of these keywords: {val_indicator_keywords}")

    try:
        subfolders = sorted([f.name for f in os.scandir(base_log_dir) if f.is_dir()])
    except FileNotFoundError:
        print(f"Error: Cannot access log directory '{base_log_dir}'. Please check the path.")
        return
        
    if not subfolders:
        print(f"No subfolders found in '{base_log_dir}'.")
        return

    for subfolder_name in subfolders:
        subfolder_path = os.path.join(base_log_dir, subfolder_name)
        try:
            log_file_names = sorted([item for item in os.listdir(subfolder_path) if item.endswith(".log")])
        except FileNotFoundError:
            print(f"Warning: Cannot access subfolder '{subfolder_path}'. Skipping.")
            continue

        for item in log_file_names:
            log_file_path = os.path.join(subfolder_path, item)
            
            log_metrics = parse_log_file(log_file_path, val_indicator_keywords, metric_preferences_config)

            if log_metrics:
                all_results.append({
                    'subfolder': subfolder_name,
                    'log_file': item,
                    'metrics': log_metrics
                })
            else:
                all_results.append({
                    'subfolder': subfolder_name,
                    'log_file': item,
                    'metrics': {} 
                })
    
    print_results_table(all_results)

def print_results_table(all_results):
    """Prints the results in a tabular format."""
    print("\n--- Detailed Log Analysis Results ---")
    if not all_results:
        print("No log files were processed, or no metrics were found.")
        return

    all_metric_keys_ordered = []
    seen_metric_keys = set()
    for res in all_results: 
        if isinstance(res['metrics'], dict):
            for k in res['metrics'].keys():
                if k not in seen_metric_keys:
                    all_metric_keys_ordered.append(k)
                    seen_metric_keys.add(k)
    
    subfolder_col_width = max(max(len(r['subfolder']) for r in all_results) if all_results else 0, len('Subfolder')) + 2
    logfile_col_width = max(max(len(r['log_file']) for r in all_results) if all_results else 0, len('Log File')) + 2
    metric_col_width = 15 

    header_parts = [f"{'Subfolder':<{subfolder_col_width}}", f"{'Log File':<{logfile_col_width}}"]
    for mk in all_metric_keys_ordered:
        header_parts.append(f"{mk:<{metric_col_width}}")
    
    header = "".join(header_parts)
    print(header)
    print("-" * len(header))

    sorted_results = sorted(all_results, key=lambda x: (x['subfolder'], x['log_file']))

    for res in sorted_results:
        row_parts = [f"{res['subfolder']:<{subfolder_col_width}}", f"{res['log_file']:<{logfile_col_width}}"]
        if isinstance(res['metrics'], dict):
            if not res['metrics']: 
                for _ in all_metric_keys_ordered: 
                    row_parts.append(f"{'N/A':<{metric_col_width}}")
            else:
                for mk in all_metric_keys_ordered:
                    val = res['metrics'].get(mk)
                    if val is None:
                        row_parts.append(f"{'N/A':<{metric_col_width}}")
                    elif isinstance(val, float):
                        row_parts.append(f"{val:<{metric_col_width}.4f}")
                    else: 
                        row_parts.append(f"{str(val):<{metric_col_width}}")
        else: 
            for _ in all_metric_keys_ordered:
                row_parts.append(f"{'Error':<{metric_col_width}}")

        print("".join(row_parts))

if __name__ == "__main__":
    main()