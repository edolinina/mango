import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def parse_evaluation_results(yaml_file):
    """Parse evaluation YAML and extract validation scores."""
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    
    results = []
    
    for task_result in data['evaluation_run']['results']:
        task = task_result['task']
        
        for agent_result in task_result['agents']:
            agent = agent_result['agent']
            
            for capability in agent_result['capabilities']:
                cap_name = capability['capability']
                validation = capability.get('validation', {})
                
                # Extract ML validator score (convert pass_rate to 1-10 scale)
                ml_validator = validation.get('ml_validator')
                ml_score = (ml_validator['pass_rate'] / 10.0) if ml_validator else None
                
                # Extract LLM judge score
                llm_judge = validation.get('llm_judge')
                llm_score = llm_judge.get('overall_score') if llm_judge else None
                
                # Extract human expert scores (average of feasibility and usefulness)
                human_expert = validation.get('human_expert', {})
                feasibility = human_expert.get('feasibility')
                usefulness = human_expert.get('usefulness')
                
                human_score = None
                if feasibility is not None and usefulness is not None:
                    human_score = (feasibility + usefulness) / 2.0
                elif feasibility is not None:
                    human_score = float(feasibility)
                elif usefulness is not None:
                    human_score = float(usefulness)
                
                results.append({
                    'task': task,
                    'agent': agent,
                    'capability': cap_name,
                    'ml_validator': ml_score,
                    'llm_judge': llm_score,
                    'human_expert': human_score
                })
    
    return pd.DataFrame(results)


def print_summary_statistics(df):
    """Print average scores for each validation method."""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY - Average Scores (1-10 scale)")
    print("="*80)
    
    # Overall averages
    print("\nOverall Averages:")
    print(f"  ML Validator:   {df['ml_validator'].mean():.2f}/10")
    print(f"  LLM Judge:      {df['llm_judge'].mean():.2f}/10")
    if df['human_expert'].notna().any():
        print(f"  Human Expert:   {df['human_expert'].mean():.2f}/10")
    
    # By task — use ordinal labels
    print("\nBy Task:")
    tasks = list(df['task'].unique())
    for idx, task in enumerate(tasks, start=1):
        task_df = df[df['task'] == task]
        print(f"\n  Task #{idx}")
        print(f"    ML Validator:   {task_df['ml_validator'].mean():.2f}/10")
        print(f"    LLM Judge:      {task_df['llm_judge'].mean():.2f}/10")
        if task_df['human_expert'].notna().any():
            print(f"    Human Expert:   {task_df['human_expert'].mean():.2f}/10")
    
    # By agent
    print("\nBy Agent:")
    for agent in df['agent'].unique():
        agent_df = df[df['agent'] == agent]
        print(f"\n  {agent}:")
        print(f"    ML Validator:   {agent_df['ml_validator'].mean():.2f}/10")
        print(f"    LLM Judge:      {agent_df['llm_judge'].mean():.2f}/10")
        if agent_df['human_expert'].notna().any():
            print(f"    Human Expert:   {agent_df['human_expert'].mean():.2f}/10")
    
    print("\n" + "="*80)


def create_visualizations(df):
    """Generate and show three plots comparing validation methods (no saving)."""
    sns.set_style("whitegrid")
    # smaller context + font scale for compact plots
    sns.set_context("paper", font_scale=0.8)
    plt.rcParams['figure.figsize'] = (8, 4)

    # 1) Overall Average Scores by Validation Method
    overall_means = df[['ml_validator', 'llm_judge', 'human_expert']].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(overall_means)), overall_means.values,
                  color=['#3498db', '#e74c3c', '#2ecc71'], width=0.6)
    ax.set_xticks(range(len(overall_means)))
    ax.set_xticklabels(['ML Validator', 'LLM Judge', 'Human Expert'], fontsize=9)
    ax.set_ylim(0, 10)
    ax.set_ylabel('Average Score (1-10)', fontsize=9)
    ax.set_title('Overall Average Scores by Validation Method', fontsize=10)
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.15, f'{h:.2f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.show()

    # 2) Average Scores by Task (use compact figure)
    task_means = df.groupby('task')[['ml_validator', 'llm_judge', 'human_expert']].mean()
    labels = [f"Task #{i}" for i in range(1, len(task_means) + 1)]
    fig, ax = plt.subplots(figsize=(8, 4))
    task_means.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c', '#2ecc71'], width=0.7)
    ax.set_ylim(0, 10)
    ax.set_ylabel('Average Score (1-10)', fontsize=9)
    ax.set_title('Average Scores by Task', fontsize=10)
    ax.legend(['ML Validator', 'LLM Judge', 'Human Expert'], fontsize=8)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
    plt.tight_layout()
    plt.show()

    # 3) Average Scores by Agent (compact)
    agent_means = df.groupby('agent')[['ml_validator', 'llm_judge', 'human_expert']].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    agent_means.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c', '#2ecc71'], width=0.7)
    ax.set_ylim(0, 10)
    ax.set_ylabel('Average Score (1-10)', fontsize=9)
    ax.set_title('Average Scores by Agent', fontsize=10)
    ax.legend(['ML Validator', 'LLM Judge', 'Human Expert'], fontsize=8)
    ax.set_xticklabels(agent_means.index, rotation=0, fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
    plt.tight_layout()
    plt.show()


def create_comparison_visualization(dfs: dict):
    """Compare scores across two models with multiple plots."""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=0.8)

    methods = ['ml_validator', 'llm_judge', 'human_expert']
    labels = ['ML Validator', 'LLM Judge', 'Human Expert']
    model_names = list(dfs.keys())
    colors = ['#3498db', '#e74c3c']
    width = 0.35

    # 1) Overall Average Scores side by side
    fig, ax = plt.subplots(figsize=(9, 4))
    x = range(len(methods))
    for i, (model_name, df) in enumerate(dfs.items()):
        means = [df[m].mean() for m in methods]
        offset = [xi + (i - 0.5) * width for xi in x]
        bars = ax.bar(offset, means, width=width, label=model_name, color=colors[i], alpha=0.85)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.15,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 10)
    ax.set_ylabel('Average Score (1-10)', fontsize=9)
    ax.set_title(f'Overall Average Scores: {" vs ".join(model_names)}', fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
    plt.tight_layout()
    plt.show()

    # 2) Per-agent comparison (llm_judge only)
    fig, axes = plt.subplots(1, len(methods), figsize=(14, 4), sharey=True)
    for ax, method, label in zip(axes, methods, labels):
        agent_means = {
            name: df.groupby('agent')[method].mean()
            for name, df in dfs.items()
        }
        agents = sorted(set().union(*[m.index for m in agent_means.values()]))
        x = range(len(agents))
        for i, (model_name, means) in enumerate(agent_means.items()):
            vals = [means.get(a, 0) for a in agents]
            offset = [xi + (i - 0.5) * width for xi in x]
            ax.bar(offset, vals, width=width, label=model_name, color=colors[i], alpha=0.85)
        ax.set_xticks(list(x))
        ax.set_xticklabels(agents, rotation=15, ha='right', fontsize=8)
        ax.set_ylim(0, 10)
        ax.set_title(label, fontsize=9)
        ax.tick_params(axis='y', labelsize=8)
        if ax == axes[0]:
            ax.set_ylabel('Average Score (1-10)', fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(f'Scores by Agent: {" vs ".join(model_names)}', fontsize=10)
    plt.tight_layout()
    plt.show()

    # 3) Per-task comparison (llm_judge)
    all_tasks = []
    for df in dfs.values():
        all_tasks.extend(df['task'].unique().tolist())
    tasks = list(dict.fromkeys(all_tasks))  # preserve order, deduplicate
    task_labels = [f"Task #{i}" for i in range(1, len(tasks) + 1)]

    fig, ax = plt.subplots(figsize=(12, 4))
    x = range(len(tasks))
    for i, (model_name, df) in enumerate(dfs.items()):
        task_means = df.groupby('task')['llm_judge'].mean()
        vals = [task_means.get(t, 0) for t in tasks]
        offset = [xi + (i - 0.5) * width for xi in x]
        ax.bar(offset, vals, width=width, label=model_name, color=colors[i], alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(task_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 10)
    ax.set_ylabel('LLM Judge Score (1-10)', fontsize=9)
    ax.set_title(f'LLM Judge Score by Task: {" vs ".join(model_names)}', fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze evaluation results')
    parser.add_argument('yaml_files', nargs='+', help='Path to one or two evaluation YAML files')
    args = parser.parse_args()

    if len(args.yaml_files) == 1:
        # single file — existing behaviour
        df = parse_evaluation_results(args.yaml_files[0])
        print_summary_statistics(df)
        create_visualizations(df)

    elif len(args.yaml_files) == 2:
        # two files — comparison mode
        dfs = {}
        for path in args.yaml_files:
            name = Path(path).stem          # e.g. "ollama" or "openai"
            dfs[name] = parse_evaluation_results(path)

        # print summary for each
        for name, df in dfs.items():
            print(f"\n{'='*80}\nModel: {name.upper()}")
            print_summary_statistics(df)

        # comparison plot
        create_comparison_visualization(dfs)

    else:
        parser.error("Provide one or two YAML files.")


if __name__ == "__main__":
    main()
