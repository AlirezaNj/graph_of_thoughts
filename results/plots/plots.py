import json
import os
import matplotlib.pyplot as plt

def get_complete_results(base_directory):
    results_complete = {}
    for folder_name in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder_name)
        if os.path.isdir(folder_path):
            results_complete[folder_name] = []
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".json"):
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        results_complete[folder_name].append(
                            {"key": int(file_name.split(".")[0]), "data": data}
                        )
        for key in results_complete.keys():
            results_complete[key] = sorted(
                results_complete[key], key=lambda x: x["key"]
            )
    return results_complete

def get_final_scores(results_complete):
    scores = {}
    for method in results_complete.keys():
        scores[method] = []
        for result in results_complete[method]:
            score = 100
            solved = False
            cost = 1
            prompt_tokens = 0
            completion_tokens = 0
            for op in result["data"]:
                if "operation" in op and op["operation"] == "ground_truth_evaluator":
                    try:
                        score = min(op["scores"])
                        solved = any(op["problem_solved"])
                    except:
                        continue
                if "cost" in op:
                    cost = op["cost"]
                    prompt_tokens = op["prompt_tokens"]
                    completion_tokens = op["completion_tokens"]
            scores[method].append(
                [result["key"], score, solved, prompt_tokens, completion_tokens, cost]
            )
        scores[method] = sorted(scores[method], key=lambda x: x[0])
    return scores

def get_final_scores_doc_merge(results_complete):
    scores = {}
    for method in results_complete.keys():
        scores[method] = []
        for result in results_complete[method]:
            score = 0
            solved = False
            cost = 1
            prompt_tokens = 0
            completion_tokens = 0
            for op in reversed(result["data"]):
                if "cost" in op:
                    cost = op["cost"]
                    prompt_tokens = op["prompt_tokens"]
                    completion_tokens = op["completion_tokens"]
                if "operation" in op and op["operation"] == "score":
                    try:
                        score = max(op["scores"])
                        break
                    except:
                        continue
            scores[method].append(
                [result["key"], score, solved, prompt_tokens, completion_tokens, cost]
            )
        scores[method] = sorted(scores[method], key=lambda x: x[0])
    return scores

def get_plotting_data(base_directory, score_method):
    results_complete = get_complete_results(base_directory)
    scores = score_method(results_complete)
    results_plotting = {
        method: {
            "scores": [x[1] for x in scores[method]],
            "solved": sum([1 for x in scores[method] if x[2]]),
            "costs": [x[5] for x in scores[method]],
        }
        for method in scores.keys()
    }
    return results_plotting

def plot_results(
    name,
    results,
    methods_order=["io", "cot", "tot", "tot2", "tog"],
    methods_labels=["IO", "CoT", "ToT", "ToT2", "GoT"],
    model="GPT-3.5",
    length=32,
    y_lower=0,
    y_upper=16,
    cost_upper=1.8,
    display_solved=True,
    annotation_offset=1,
    display_left_ylabel=False,
    display_right_ylabel=False,
    save_path=None,
    left_axis_max=None,
    left_axis_tick_step=None,
    right_axis_max=None,
    right_axis_tick_step=None,
    figsize_override=None,
    xtick_rotation=0,
):
    methods_order = [method for method in methods_order if method in results]
    # Extract scores based on the order
    if name == "set_intersection":
        scores_ordered = [
            [min(score, length) for score in results[method]["scores"] if score != 1000]
            for method in methods_order
        ]
    elif name == "sorting":
        scores_ordered = [
            [
                min(score, length)
                for score in results[method]["scores"]
                if score != 100 and score != 300
            ]
            for method in methods_order
        ]
    elif name == "keyword_counting":
        scores_ordered = [
            [
                score
                for score in results[method]["scores"]
                if score != 100 and score != 300
            ]
            for method in methods_order
        ]
    elif name == "document_merging":
        scores_ordered = [
            [score for score in results[method]["scores"]] for method in methods_order
        ]
    total_costs = [sum(results[method]["costs"]) for method in methods_order]

    # Create figure and axis
    if figsize_override is not None:
        fig, ax = plt.subplots(dpi=150, figsize=figsize_override)
    elif name == "keyword_counting" or name == "document_merging":
        fig, ax = plt.subplots(dpi=150, figsize=(3.75, 5))
    else:
        fig, ax = plt.subplots(dpi=150, figsize=(2.5, 5))

    positions = range(1, len(methods_order) + 1)
    # If every method has only one score (e.g. from CSV), use bars so it doesn't look like flat lines
    single_value_per_method = all(len(s) == 1 for s in scores_ordered)
    if single_value_per_method and name == "sorting":
        means = [s[0] if s else 0 for s in scores_ordered]
        ax.bar([p - 0.2 for p in positions], means, width=0.35, color="C0", align="center")
    else:
        ax.boxplot(scores_ordered, positions=positions)

    fig_fontsize = 12

    # Set the ticks and labels
    plt.yticks(fontsize=fig_fontsize)
    ax.set_xticks(range(1, len(methods_order) + 1))
    ax.set_xticks(range(1, len(methods_order) + 1))
    if name == "keyword_counting":
        ax.set_xticklabels(methods_labels, fontsize=10, rotation=xtick_rotation)
    else:
        ax.set_xticklabels(methods_labels, fontsize=fig_fontsize, rotation=xtick_rotation)

    if name == "document_merging":
        ax.set_ylim(y_lower, 12 if display_solved else 9.75)
    else:
        if left_axis_max is not None:
            ax.set_ylim(y_lower, left_axis_max)
        else:
            ax.set_ylim(y_lower, (y_upper + 2) if display_solved else y_upper + 1)

    if name == "sorting" or name == "set_intersection":
        if left_axis_max is not None and left_axis_tick_step is not None:
            ax.set_yticks(list(range(y_lower, left_axis_max + 1, left_axis_tick_step)))
        else:
            ax1_yticks = range(
                y_lower, y_upper + 1, 2 if length < 48 else (4 if length < 96 else 8)
            )
            ax.set_yticks(ax1_yticks)

    if display_left_ylabel:
        if name == "keyword_counting":
            ax.set_ylabel(
                f"Number of errors; the lower the better", fontsize=fig_fontsize
            )
        elif name == "document_merging":
            ax.set_ylabel(
                f"Score (out of 10); the higher the better", fontsize=fig_fontsize
            )
        else:
            ax.set_ylabel(
                f"#incorrect elements; the lower the better", fontsize=fig_fontsize
            )

    if name == "sorting" or name == "set_intersection":
        ax.set_title(f"{length} elements")

    ax2 = ax.twinx()
    if single_value_per_method and name == "sorting":
        ax2.bar([p + 0.2 for p in positions], total_costs, width=0.35, alpha=0.5, color="blue", label="Total Cost ($)")
    else:
        ax2.bar(positions, total_costs, alpha=0.5, color="blue", label="Total Cost ($)")
    ax2.yaxis.set_tick_params(colors="#1919ff", labelsize=fig_fontsize)
    if right_axis_max is not None and right_axis_tick_step is not None:
        ax2.set_ylim(0, right_axis_max)
        ax2.set_yticks([round(i * right_axis_tick_step, 2) for i in range(int(round(right_axis_max / right_axis_tick_step)) + 1)])
    else:
        ax2.set_ylim(0, cost_upper)
        number_of_ticks = len(ax.get_yticks())
        tick_interval = cost_upper / (number_of_ticks)
        ax2_ticks = [tick_interval * i for i in range(number_of_ticks)]
        ax2.set_yticks(ax2_ticks)

    if display_right_ylabel:
        ax2.set_ylabel(
            "Total Cost ($); the lower the better",
            color="#1919ff",
            fontsize=fig_fontsize,
        )

    if display_solved:
        annotation_height = (left_axis_max if left_axis_max is not None else y_upper) + annotation_offset
        count = 1
        for method in methods_order:
            if method not in results:
                continue
            solved = results[method]["solved"]
            ax.text(
                count,
                annotation_height,
                f"{solved}",
                ha="center",
                va="bottom",
                fontsize=fig_fontsize,
            )
            count += 1

    model_sanitized = model.replace(".", "").replace("-", "").lower()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    elif name == "keyword_counting" or name == "document_merging":
        fig.savefig(f"{name}_{model_sanitized}.png", bbox_inches="tight")
    else:
        fig.savefig(f"{name}_{model_sanitized}_{length}.png", bbox_inches="tight")


def plot_sorting_from_csv(
    method_stats,
    methods_order=("io", "cot", "tot", "tot2", "got"),
    methods_labels=("IO", "CoT", "ToT", "ToT2", "GoT"),
    model_label="",
    length=32,
    y_upper=16,
    cost_upper=1.8,
    save_path=None,
):
    """
    Plot sorting results from aggregated CSV: bar of mean errors (left),
    bar of total cost (right), and annotation of success count per method.
    method_stats: list of dicts with keys "method", "score_avg", "cost_total", "solved"
    """
    method_stats = [m for m in method_stats if m["method"] in methods_order]
    methods_present = [m["method"] for m in method_stats]
    if not methods_present:
        return
    scores = [next(m["score_avg"] for m in method_stats if m["method"] == meth) for meth in methods_present]
    total_costs = [next(m["cost_total"] for m in method_stats if m["method"] == meth) for meth in methods_present]
    solved_counts = [next(m["solved"] for m in method_stats if m["method"] == meth) for meth in methods_present]
    labels = [methods_labels[methods_order.index(m)] for m in methods_present]

    fig, ax = plt.subplots(dpi=150, figsize=(2.5, 5))
    positions = range(1, len(methods_present) + 1)
    ax.bar([p - 0.2 for p in positions], scores, 0.35, color="C0", label="# errors")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("#incorrect elements; the lower the better", fontsize=12)
    ax.set_ylim(0, min(y_upper, max(scores) * 1.2 + 2) if scores else y_upper)
    ax.set_title(f"{length} elements" + (f" ({model_label})" if model_label else ""))

    ax2 = ax.twinx()
    ax2.bar([p + 0.2 for p in positions], total_costs, 0.35, alpha=0.5, color="blue")
    ax2.yaxis.set_tick_params(colors="#1919ff", labelsize=12)
    ax2.set_ylim(0, max(total_costs) * 1.15 + 0.01 if total_costs else cost_upper)
    ax2.set_ylabel("Total Cost ($); the lower the better", color="#1919ff", fontsize=12)

    for i, s in enumerate(solved_counts):
        ax.text(i + 1, min(y_upper, (max(scores) * 1.1 + 1) if scores else y_upper) + 0.5, f"{s}", ha="center", va="bottom", fontsize=12)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_results(
        "set_intersection",
        get_plotting_data("set_intersection_gpt35_032", get_final_scores),
        methods_order=["io", "cot", "tot", "tot2", "tog2"],
        length=32,
        y_upper=19,
        cost_upper=2,
        display_solved=True,
        annotation_offset=0.5,
        display_left_ylabel=True,
        display_right_ylabel=True,
    )

    plot_results(
        "set_intersection",
        get_plotting_data("set_intersection_gpt35_064", get_final_scores),
        methods_order=["io", "cot", "tot", "tot2", "tog2"],
        length=64,
        y_upper=32,
        cost_upper=5.4,
        display_solved=True,
        annotation_offset=0.2,
        display_left_ylabel=True,
        display_right_ylabel=True,
    )

    plot_results(
        "set_intersection",
        get_plotting_data("set_intersection_gpt35_128", get_final_scores),
        methods_order=["io", "cot", "tot", "tot2", "tog2"],
        length=128,
        y_upper=94,
        cost_upper=12,
        display_solved=True,
        annotation_offset=-3,
        display_left_ylabel=True,
        display_right_ylabel=True,
    )

    plot_results(
        "sorting",
        get_plotting_data("sorting_gpt35_032", get_final_scores),
        length=32,
        y_upper=16,
        cost_upper=1.8,
        display_solved=True,
        annotation_offset=0.5,
        display_left_ylabel=True,
        display_right_ylabel=True,
        left_axis_max=None,
        left_axis_tick_step=2,
        right_axis_max=None,
        right_axis_tick_step=0.2,
    )

    plot_results(
        "sorting",
        get_plotting_data("sorting_gpt35_064", get_final_scores),
        length=64,
        y_upper=64,
        cost_upper=5.1,
        display_solved=True,
        display_left_ylabel=True,
        display_right_ylabel=True,
    )

    plot_results(
        "sorting",
        get_plotting_data("sorting_gpt35_128", get_final_scores),
        length=128,
        y_upper=128,
        cost_upper=17,
        display_solved=True,
        display_left_ylabel=True,
        display_right_ylabel=True,
    )

    plot_results(
        "keyword_counting",
        get_plotting_data("keyword_counting_gpt35", get_final_scores),
        methods_order=["io", "cot", "tot", "tot2", "gsp4", "gsp8", "gspx"],
        methods_labels=["IO", "CoT", "ToT", "ToT2", "GoT4", "GoT8", "GoTx"],
        y_upper=35,
        cost_upper=9,
        display_solved=True,
        annotation_offset=-0.3,
        display_left_ylabel=True,
        display_right_ylabel=True,
    )

    plot_results(
        "document_merging",
        get_plotting_data("document_merging_gpt35_16k", get_final_scores_doc_merge),
        methods_order=["io", "cot", "tot", "gsp", "gsp2"],
        methods_labels=["IO", "CoT", "ToT", "GoT", "GoT2"],
        y_upper=10,
        cost_upper=15,
        display_solved=False,
        display_left_ylabel=True,
        display_right_ylabel=True,
    )