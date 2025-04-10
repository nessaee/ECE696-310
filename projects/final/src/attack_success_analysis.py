"""
Attack Success Analysis for HarmBench Dataset

This script analyzes the HarmBench dataset to measure the probability of successful 
attack at each turn for each attack tactic. It models the conversation as a binary tree
where each layer represents an attack turn, and the tree terminates if an attack is successful.
"""

import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional
import sys

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from harmbench_data import HarmBenchDataset, QuestionBehavior, BehaviorEntry, MessageEntry

class AttackSuccessAnalyzer:
    """
    Analyzes the success rate of different attack tactics at each turn in a conversation.
    
    Models the conversation as a binary tree where each layer represents an attack turn,
    and the tree terminates if an attack is successful.
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize the analyzer with the path to the HarmBench dataset.
        
        Args:
            dataset_path: Path to the HarmBench behaviors CSV file.
        """
        self.dataset = HarmBenchDataset()
        self.dataset.load_from_csv(dataset_path)
        
        # Dictionary to store success indicators by tactic and turn
        self.success_by_tactic_turn = defaultdict(lambda: defaultdict(list))
        
        # Set of tactics found in the dataset
        self.tactics = set()
        
        # Maximum number of turns across all conversations
        self.max_turns = 0

    def analyze_conversation_turns(self, entry: BehaviorEntry) -> int:
        """
        Analyze the conversation to determine at which turn the attack succeeded.
        
        Args:
            entry: A behavior entry from the dataset.
            
        Returns:
            The turn number at which the attack succeeded, or -1 if there are no user messages.
        """
        # Count user messages as turns (excluding system prompt which is now stored separately)
        user_messages = [msg for msg in entry.messages if msg.role == "user" and msg.body]
        
        # If there are no user messages, can't determine turn
        if not user_messages:
            return -1
        
        # As per requirement, assume the attack was always successful at the last message
        # In a more sophisticated analysis, you would analyze each assistant response
        # to determine exactly when the attack succeeded
        return len(user_messages)
    
    def analyze_dataset(self):
        """
        Analyze the entire dataset to compute success rates by tactic and turn.
        """
        # Process each question and entry
        for question_id, question in self.dataset.questions.items():
            for temp, entry in question.entries.items():
                # Track the tactic
                self.tactics.add(entry.tactic)
                
                # Determine at which turn the attack succeeded
                success_turn = self.analyze_conversation_turns(entry)
                
                # Record the success at the final turn
                if success_turn > 0:
                    # Mark all turns before the last as unsuccessful
                    for turn in range(1, success_turn):
                        self.success_by_tactic_turn[entry.tactic][turn].append(0)
                    
                    # Mark the last turn as successful
                    self.success_by_tactic_turn[entry.tactic][success_turn].append(1)
                    
                    # Update max turns if needed
                    self.max_turns = max(self.max_turns, success_turn)
    
    def calculate_success_probabilities(self) -> Dict[str, List[float]]:
        """
        Calculate the probability of successful attack at each turn for each tactic.
        
        Returns:
            A dictionary mapping tactics to lists of success probabilities by turn.
        """
        probabilities = {}
        
        for tactic in self.tactics:
            tactic_probs = []
            
            for turn in range(1, self.max_turns + 1):
                # Get success indicators for this tactic and turn
                success_indicators = self.success_by_tactic_turn[tactic][turn]
                
                # Calculate probability if there are any data points
                if success_indicators:
                    prob = sum(success_indicators) / len(success_indicators)
                else:
                    prob = 0.0
                
                tactic_probs.append(prob)
            
            probabilities[tactic] = tactic_probs
        
        return probabilities
    
    def calculate_cumulative_success_probabilities(self) -> Dict[str, List[float]]:
        """
        Calculate the cumulative probability of successful attack by each turn for each tactic.
        
        Returns:
            A dictionary mapping tactics to lists of cumulative success probabilities by turn.
        """
        turn_probs = self.calculate_success_probabilities()
        cumulative_probs = {}
        
        for tactic, probs in turn_probs.items():
            # Calculate cumulative probability (1 - probability of failing at all turns)
            cumulative = []
            cumulative_fail_prob = 1.0
            
            for prob in probs:
                cumulative_fail_prob *= (1.0 - prob)
                cumulative.append(1.0 - cumulative_fail_prob)
            
            cumulative_probs[tactic] = cumulative
        
        return cumulative_probs
    
    def plot_success_probabilities(self, save_path: Optional[str] = None):
        """
        Plot the probability of successful attack at each turn for each tactic.
        
        Args:
            save_path: Optional path to save the plot. If None, the plot is displayed.
        """
        probabilities = self.calculate_success_probabilities()
        
        plt.figure(figsize=(12, 8))
        
        for tactic, probs in probabilities.items():
            plt.plot(range(1, len(probs) + 1), probs, marker='o', label=tactic)
        
        plt.title('Probability of Successful Attack at Each Turn by Tactic')
        plt.xlabel('Turn Number')
        plt.ylabel('Success Probability')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Set consistent y-axis scaling from 0 to 1
        plt.ylim(0, 1.0)
        
        # Set x-axis ticks to integers
        plt.xticks(range(1, self.max_turns + 1))
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()
            
    def plot_success_bar_charts(self, output_dir: str):
        """
        Create bar charts showing attack success rates by turn for each tactic.
        Compiles all charts into a single figure with multiple subplots.
        
        Args:
            output_dir: Directory to save the bar chart visualizations.
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get success probabilities
        probabilities = self.calculate_success_probabilities()
        
        # Determine the number of tactics and create a grid layout for subplots
        num_tactics = len(probabilities)
        if num_tactics == 0:
            return
            
        # Calculate grid dimensions for subplots
        # Aim for a roughly square layout
        grid_cols = int(np.ceil(np.sqrt(num_tactics)))
        grid_rows = int(np.ceil(num_tactics / grid_cols))
        
        # Create figure with subplots
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(15, 10))
        fig.suptitle('Attack Success Probability by Turn for Each Tactic', fontsize=16)
        
        # Flatten axes array for easier indexing if there are multiple rows and columns
        if grid_rows > 1 and grid_cols > 1:
            axes = axes.flatten()
        elif grid_rows == 1 and grid_cols > 1:
            axes = np.array([axes])
            axes = axes.flatten()
        elif grid_rows > 1 and grid_cols == 1:
            axes = np.array([[ax] for ax in axes])
            axes = axes.flatten()
        else:
            # If there's only one subplot, wrap it in a list for consistent indexing
            axes = np.array([axes])
        
        # Create a bar chart for each tactic in its own subplot
        for i, (tactic, probs) in enumerate(sorted(probabilities.items())):
            if i < len(axes):  # Ensure we don't exceed the number of subplots
                ax = axes[i]
                
                # Create bar chart
                turns = range(1, len(probs) + 1)
                ax.bar(turns, probs, color='skyblue', alpha=0.7)
                
                # Add title and labels
                ax.set_title(tactic)
                ax.set_xlabel('Turn Number')
                ax.set_ylabel('Success Probability')
                ax.set_ylim(0, 1.0)  # Consistent y-axis scale across all plots
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                ax.set_xticks(turns)
        
        # Hide any unused subplots
        for j in range(num_tactics, len(axes)):
            axes[j].set_visible(False)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
        
        # Save the combined figure
        combined_path = os.path.join(output_dir, "all_tactics_bar_charts.png")
        plt.savefig(combined_path, dpi=300)
        plt.close()
        
        # Create a grouped bar chart for all tactics in a single plot
        plt.figure(figsize=(14, 8))
        
        # Determine the maximum number of turns across all tactics
        max_turns = max(len(probs) for probs in probabilities.values())
        
        # Set width of bars based on number of tactics
        bar_width = 0.8 / num_tactics if num_tactics > 0 else 0.8
        
        # Create grouped bar chart
        for i, (tactic, probs) in enumerate(sorted(probabilities.items())):
            # Calculate position for this group of bars
            positions = [j + (i - num_tactics/2 + 0.5) * bar_width for j in range(1, len(probs) + 1)]
            plt.bar(positions, probs, width=bar_width, label=tactic, alpha=0.7)
        
        # Add title and labels
        plt.title('Comparison of Attack Success Probability by Turn for All Tactics')
        plt.xlabel('Turn Number')
        plt.ylabel('Success Probability')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(range(1, max_turns + 1))
        plt.legend()
        
        # Save the grouped chart
        grouped_path = os.path.join(output_dir, "grouped_tactics_bar_chart.png")
        plt.savefig(grouped_path, dpi=300)
        plt.close()
    
    def plot_cumulative_success_probabilities(self, save_path: Optional[str] = None):
        """
        Plot the cumulative probability of successful attack by each turn for each tactic.
        
        Args:
            save_path: Optional path to save the plot. If None, the plot is displayed.
        """
        cumulative_probs = self.calculate_cumulative_success_probabilities()
        
        plt.figure(figsize=(12, 8))
        
        for tactic, probs in cumulative_probs.items():
            plt.plot(range(1, len(probs) + 1), probs, marker='o', label=tactic)
        
        plt.title('Cumulative Probability of Successful Attack by Turn for Each Tactic')
        plt.xlabel('Turn Number')
        plt.ylabel('Cumulative Success Probability')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Set consistent y-axis scaling from 0 to 1
        plt.ylim(0, 1.0)
        
        # Set x-axis ticks to integers
        plt.xticks(range(1, self.max_turns + 1))
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()
    
    def visualize_attack_tree(self, tactic: str, save_path: Optional[str] = None):
        """
        Visualize the attack tree for a specific tactic, including all turns available in the dataset.
        
        Args:
            tactic: The tactic to visualize.
            save_path: Optional path to save the visualization. If None, the plot is displayed.
        """
        # Get success probabilities for this tactic
        probabilities = self.calculate_success_probabilities()[tactic]
        
        # Count the number of turns available for this tactic
        turn_counts = []
        for question_id, question in self.dataset.questions.items():
            for temp, entry in question.entries.items():
                if entry.tactic == tactic:
                    user_messages = [msg for msg in entry.messages if msg.role == "user" and msg.body]
                    turn_count = len(user_messages)
                    if turn_count > 0:
                        turn_counts.append(turn_count)
        
        # Get the maximum turn count for this tactic
        total_conversations = len(turn_counts)
        max_turns = max(turn_counts) if turn_counts else 0
        
        # Only use up to the maximum turn count for this tactic
        depth = min(max_turns, len(probabilities))
        probs = probabilities[:depth]
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Calculate horizontal spacing based on depth
        # More depth = more spacing to prevent overlap
        h_spacing = max(2.0, depth * 0.5)  # Increase spacing for deeper trees
        
        # Add the root node with turn count information
        G.add_node("root", pos=(0, 0), label=f"Start\nConvs: {total_conversations}\nMax: {max_turns} turns")
        
        # Add nodes for each level with increased spacing
        for level in range(1, depth + 1):
            # Success node for this level
            success_node = f"s{level}"
            G.add_node(success_node, pos=(level * h_spacing, 2), label=f"{level}S")
            
            # Only add failure nodes for non-final levels
            if level < depth:
                # Failure node for this level
                failure_node = f"f{level}"
                G.add_node(failure_node, pos=(level * h_spacing, -2), label=f"{level}F")
            
            # Connect from previous level
            if level == 1:
                # Connect from root to first level
                G.add_edge("root", success_node, weight=probs[level-1])
                if level < depth:  # Only connect to failure if not the last level
                    G.add_edge("root", failure_node, weight=1-probs[level-1])
            else:
                # Connect from previous failure node
                prev_failure = f"f{level-1}"
                G.add_edge(prev_failure, success_node, weight=probs[level-1])
                if level < depth:  # Only connect to failure if not the last level
                    G.add_edge(prev_failure, failure_node, weight=1-probs[level-1])
        
        # Create the plot
        plt.figure(figsize=(14, 10))
        
        # Get node positions
        pos = nx.get_node_attributes(G, 'pos')
        
        # Use consistent node size for all nodes
        node_size = 1000
        
        # Draw all nodes with the same size
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='lightblue', alpha=0.8)
        
        # Draw edges with consistent width but different styles based on type
        success_edges = [(u, v) for u, v in G.edges() if v.startswith('s')]
        failure_edges = [(u, v) for u, v in G.edges() if v.startswith('f')]
        
        # Draw success edges
        nx.draw_networkx_edges(G, pos, edgelist=success_edges, width=2, alpha=0.8, 
                              edge_color='green', connectionstyle='arc3,rad=0.1')
        
        # Draw failure edges
        nx.draw_networkx_edges(G, pos, edgelist=failure_edges, width=2, alpha=0.8, 
                              edge_color='red', connectionstyle='arc3,rad=0.1')
        
        # Draw node labels
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
        
        # Draw edge labels (probabilities) with better positioning
        edge_labels = {(u, v): f"{data['weight']:.2f}" for u, v, data in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, 
                                    label_pos=0.3, bbox=dict(boxstyle="round,pad=0.3", 
                                                            fc="white", ec="none", alpha=0.7))
        
        # Add title and adjust layout
        plt.title(f"Attack Tree for Tactic: {tactic} (Total: {len(turn_counts)} conversations)")
        plt.axis('off')
        plt.tight_layout()
        
        # Save or display the plot
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()
    
    def visualize_all_attack_trees(self, output_dir: str):
        """
        Visualize attack trees for all tactics and save them to the specified directory.
        Each tree will include all turns available in the dataset for that tactic.
        
        Args:
            output_dir: Directory to save the visualizations.
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Visualize tree for each tactic
        for tactic in sorted(self.tactics):
            # Create a valid filename
            filename = f"{tactic.replace(' ', '_').lower()}_tree.png"
            save_path = os.path.join(output_dir, filename)
            
            # Visualize and save the tree with all available turns
            self.visualize_attack_tree(tactic, save_path)
    
    def generate_report(self):
        """
        Generate a comprehensive report on attack success rates by tactic and turn.
        """
        probabilities = self.calculate_success_probabilities()
        cumulative_probs = self.calculate_cumulative_success_probabilities()
        
        print("=" * 80)
        print("ATTACK SUCCESS ANALYSIS REPORT")
        print("=" * 80)
        
        print("\nTactics found in the dataset:")
        for tactic in sorted(self.tactics):
            print(f"- {tactic}")
        
        print("\nSuccess Probabilities by Turn and Tactic:")
        print("-" * 80)
        
        # Print header row
        header = "Tactic"
        for turn in range(1, self.max_turns + 1):
            header += f" | Turn {turn}"
        print(header)
        print("-" * len(header))
        
        # Print data rows
        for tactic in sorted(self.tactics):
            row = tactic
            for turn in range(self.max_turns):
                if turn < len(probabilities[tactic]):
                    row += f" | {probabilities[tactic][turn]:.3f}"
                else:
                    row += " | N/A"
            print(row)
        
        print("\nCumulative Success Probabilities by Turn and Tactic:")
        print("-" * 80)
        
        # Print header row
        header = "Tactic"
        for turn in range(1, self.max_turns + 1):
            header += f" | Turn {turn}"
        print(header)
        print("-" * len(header))
        
        # Print data rows
        for tactic in sorted(self.tactics):
            row = tactic
            for turn in range(self.max_turns):
                if turn < len(cumulative_probs[tactic]):
                    row += f" | {cumulative_probs[tactic][turn]:.3f}"
                else:
                    row += " | N/A"
            print(row)
        
        print("\nSummary of Findings:")
        print("-" * 80)
        
        # Find most and least effective tactics
        final_probs = {tactic: probs[-1] for tactic, probs in cumulative_probs.items() if probs}
        
        if final_probs:
            most_effective = max(final_probs.items(), key=lambda x: x[1])
            least_effective = min(final_probs.items(), key=lambda x: x[1])
            
            print(f"Most effective tactic: {most_effective[0]} with {most_effective[1]:.3f} cumulative success probability")
            print(f"Least effective tactic: {least_effective[0]} with {least_effective[1]:.3f} cumulative success probability")
        
        # Calculate average turns to success for each tactic
        print("\nAverage Turns to Success by Tactic:")
        for tactic in sorted(self.tactics):
            success_turns = []
            
            for question_id, question in self.dataset.questions.items():
                for temp, entry in question.entries.items():
                    if entry.tactic == tactic:
                        success_turn = self.analyze_conversation_turns(entry)
                        if success_turn > 0:
                            success_turns.append(success_turn)
            
            if success_turns:
                avg_turns = sum(success_turns) / len(success_turns)
                print(f"- {tactic}: {avg_turns:.2f} turns (from {len(success_turns)} successful attacks)")
            else:
                print(f"- {tactic}: No successful attacks")


    def plot_turn_count_histograms(self, output_dir: str):
        """
        Create histograms showing the distribution of turn counts for each tactic.
        
        Args:
            output_dir: Directory to save the histogram visualizations.
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Dictionary to store turn counts by tactic
        turn_counts_by_tactic = defaultdict(list)
        
        # Collect turn counts for each tactic
        for question_id, question in self.dataset.questions.items():
            for temp, entry in question.entries.items():
                tactic = entry.tactic
                # Count only user messages (excluding system prompt which is now separate)
                user_messages = [msg for msg in entry.messages if msg.role == "user" and msg.body]
                turn_count = len(user_messages)
                if turn_count > 0:  # Only include conversations with at least one turn
                    turn_counts_by_tactic[tactic].append(turn_count)
        
        # Calculate grid dimensions for subplots
        num_tactics = len(turn_counts_by_tactic)
        if num_tactics == 0:
            return
            
        grid_cols = int(np.ceil(np.sqrt(num_tactics)))
        grid_rows = int(np.ceil(num_tactics / grid_cols))
        
        # Create figure with subplots
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(15, 10))
        fig.suptitle('Distribution of Turn Counts by Tactic', fontsize=16)
        
        # Flatten axes array for easier indexing
        if grid_rows > 1 and grid_cols > 1:
            axes = axes.flatten()
        elif grid_rows == 1 and grid_cols > 1:
            axes = np.array([axes])
            axes = axes.flatten()
        elif grid_rows > 1 and grid_cols == 1:
            axes = np.array([[ax] for ax in axes])
            axes = axes.flatten()
        else:
            axes = np.array([axes])
        
        # Find global max frequency for consistent y-axis scaling
        max_frequency = 0
        max_turn_count = 0
        for tactic, counts in turn_counts_by_tactic.items():
            if counts:
                # Get histogram data without plotting
                hist_values, _ = np.histogram(counts, bins=range(1, max(counts) + 2))
                max_frequency = max(max_frequency, max(hist_values))
                max_turn_count = max(max_turn_count, max(counts))
        
        # Create a histogram for each tactic
        for i, (tactic, counts) in enumerate(sorted(turn_counts_by_tactic.items())):
            if i < len(axes):
                ax = axes[i]
                
                # Create histogram
                ax.hist(counts, bins=range(1, max_turn_count + 2), alpha=0.7, color='skyblue', edgecolor='black')
                
                # Add statistics
                mean_turns = np.mean(counts)
                median_turns = np.median(counts)
                ax.axvline(mean_turns, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_turns:.2f}')
                ax.axvline(median_turns, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_turns:.2f}')
                
                # Add title and labels
                ax.set_title(tactic)
                ax.set_xlabel('Number of Turns')
                ax.set_ylabel('Frequency')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                ax.legend()
                
                # Set consistent axis limits
                ax.set_xlim(0, max_turn_count + 1)
                ax.set_ylim(0, max_frequency * 1.1)  # Add 10% padding
        
        # Hide any unused subplots
        for j in range(num_tactics, len(axes)):
            axes[j].set_visible(False)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, "turn_count_histograms.png"), dpi=300)
        plt.close()
        
        # Create a single combined histogram with all tactics
        plt.figure(figsize=(12, 8))
        
        # Determine max turns across all tactics for consistent binning
        all_counts = [count for counts in turn_counts_by_tactic.values() for count in counts]
        max_turns = max(all_counts) if all_counts else 0
        
        # Plot histogram for each tactic with transparency
        for tactic, counts in sorted(turn_counts_by_tactic.items()):
            plt.hist(counts, bins=range(1, max_turns + 2), alpha=0.5, label=tactic)
        
        plt.title('Distribution of Turn Counts Across All Tactics')
        plt.xlabel('Number of Turns')
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        
        # Set consistent x-axis limits
        plt.xlim(0, max_turns + 1)
        
        # Save the combined histogram
        plt.savefig(os.path.join(output_dir, "combined_turn_count_histogram.png"), dpi=300)
        plt.close()


if __name__ == "__main__":
    # Path to the HarmBench behaviors CSV file
    dataset_path = "dataset/harmbench_behaviors.csv"
    
    # Create and initialize the analyzer
    analyzer = AttackSuccessAnalyzer(dataset_path)
    
    # Analyze the dataset
    analyzer.analyze_dataset()
    
    # Generate the report
    analyzer.generate_report()
    
    # Plot success probabilities
    analyzer.plot_success_probabilities("attack_success_by_turn.png")
    
    # Plot cumulative success probabilities
    analyzer.plot_cumulative_success_probabilities("cumulative_attack_success.png")
    
    # Create bar charts for attack success by turn
    analyzer.plot_success_bar_charts("bar_charts")
    
    # Create histograms of turn counts for each tactic
    analyzer.plot_turn_count_histograms("histograms")
    
    # Visualize attack trees for all tactics
    analyzer.visualize_all_attack_trees("attack_trees")
    
    print(f"\nPlots saved to 'attack_success_by_turn.png' and 'cumulative_attack_success.png'")
    print(f"Bar charts saved to directory: 'bar_charts'")
    print(f"Histograms saved to directory: 'histograms'")
    print(f"Attack trees saved to directory: 'attack_trees'")
