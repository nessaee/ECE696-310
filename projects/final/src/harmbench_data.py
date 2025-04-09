"""
HarmBench Behaviors Data Structure

This module provides a data structure to represent the HarmBench behaviors dataset.
Each question ID is initialized as an object that may have entries for different
temperature settings (0.0, 0.5, 1.0), and includes attributes such as source,
tactic, time_spent, message sequence, and sequence length.
"""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class MessageEntry:
    """Represents a single message in a conversation sequence."""
    role: str
    body: str

    @classmethod
    def from_json_str(cls, json_str: str) -> 'MessageEntry':
        """Create a MessageEntry from a JSON string representation."""
        import json
        try:
            data = json.loads(json_str)
            return cls(role=data.get('role', ''), body=data.get('body', ''))
        except (json.JSONDecodeError, KeyError):
            # Return empty message if parsing fails
            return cls(role='', body='')


@dataclass
class BehaviorEntry:
    """Represents a single behavior entry at a specific temperature."""
    source: str
    temperature: float
    tactic: str
    time_spent: int
    submission_message: str
    messages: List[MessageEntry] = field(default_factory=list)
    sequence_length: int = 0

    def __post_init__(self):
        """Calculate sequence length after initialization."""
        self.sequence_length = len([msg for msg in self.messages if msg.role and msg.body])


@dataclass
class QuestionBehavior:
    """
    Represents behaviors for a specific question ID across different temperatures.
    
    Each question may have entries for temperatures 0.0, 0.5, and 1.0.
    """
    question_id: str
    entries: Dict[float, BehaviorEntry] = field(default_factory=dict)

    def add_entry(self, entry: BehaviorEntry) -> None:
        """Add a behavior entry for a specific temperature."""
        self.entries[entry.temperature] = entry

    def get_entry(self, temperature: float) -> Optional[BehaviorEntry]:
        """Get the behavior entry for a specific temperature if it exists."""
        return self.entries.get(temperature)


class HarmBenchDataset:
    """
    Represents the entire HarmBench behaviors dataset.
    
    Provides methods to load data from CSV, access question behaviors,
    and perform basic analysis on the dataset.
    """
    def __init__(self):
        self.questions: Dict[str, QuestionBehavior] = {}
    
    def load_from_csv(self, csv_path: str) -> None:
        """
        Load HarmBench behaviors data from a CSV file.
        
        Args:
            csv_path: Path to the CSV file containing HarmBench behaviors data.
        """
        import csv
        
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader)  # Get the header row
            
            # Find indices for message columns
            message_indices = {}
            for i, col in enumerate(header):
                if col.startswith('message_'):
                    message_indices[i] = int(col.split('_')[1])
            
            # Find indices for required columns
            source_idx = header.index('Source')
            temp_idx = header.index('temperature')
            tactic_idx = header.index('tactic')
            question_id_idx = header.index('question_id')
            time_spent_idx = header.index('time_spent')
            submission_msg_idx = header.index('submission_message')
            
            # Process each row
            for row in reader:
                if not row:
                    continue
                    
                question_id = row[question_id_idx]
                temperature = float(row[temp_idx])
                
                # Create message entries
                messages = []
                for idx, msg_num in message_indices.items():
                    if idx < len(row) and row[idx]:
                        messages.append(MessageEntry.from_json_str(row[idx]))
                    else:
                        messages.append(MessageEntry(role='', body=''))
                
                # Create behavior entry
                entry = BehaviorEntry(
                    source=row[source_idx],
                    temperature=temperature,
                    tactic=row[tactic_idx],
                    time_spent=int(row[time_spent_idx]) if row[time_spent_idx].isdigit() else 0,
                    submission_message=row[submission_msg_idx],
                    messages=messages
                )
                
                # Add to dataset
                if question_id not in self.questions:
                    self.questions[question_id] = QuestionBehavior(question_id=question_id)
                
                self.questions[question_id].add_entry(entry)
    
    def get_question(self, question_id: str) -> Optional[QuestionBehavior]:
        """Get the behavior data for a specific question ID if it exists."""
        return self.questions.get(question_id)
    
    def get_questions_by_tactic(self, tactic: str) -> List[QuestionBehavior]:
        """Get all questions that use a specific tactic."""
        result = []
        for question in self.questions.values():
            for entry in question.entries.values():
                if entry.tactic == tactic:
                    result.append(question)
                    break
        return result
    
    def get_questions_by_temperature(self, temperature: float) -> List[QuestionBehavior]:
        """Get all questions that have entries for a specific temperature."""
        return [q for q in self.questions.values() if temperature in q.entries]
    
    def get_average_time_spent(self) -> float:
        """Calculate the average time spent across all entries."""
        total_time = 0
        count = 0
        
        for question in self.questions.values():
            for entry in question.entries.values():
                total_time += entry.time_spent
                count += 1
        
        return total_time / count if count > 0 else 0
    
    def get_average_sequence_length(self) -> float:
        """Calculate the average sequence length across all entries."""
        total_length = 0
        count = 0
        
        for question in self.questions.values():
            for entry in question.entries.values():
                total_length += entry.sequence_length
                count += 1
        
        return total_length / count if count > 0 else 0


# Example usage
if __name__ == "__main__":
    # Path to the HarmBench behaviors CSV file
    csv_path = "dataset/harmbench_behaviors.csv"
    
    # Create and load the dataset
    dataset = HarmBenchDataset()
    dataset.load_from_csv(csv_path)
    
    # Print some basic statistics
    print(f"Total questions: {len(dataset.questions)}")
    print(f"Average time spent: {dataset.get_average_time_spent():.2f}")
    print(f"Average sequence length: {dataset.get_average_sequence_length():.2f}")
    
    # Example: Access a specific question and temperature
    question_id = "7"  # Example question ID
    for i in range(50):
        question_id = str(i)

        if question_id in dataset.questions:
            question = dataset.questions[question_id]
            temp_0_entry = question.get_entry(0.0)
            if temp_0_entry:
                print(f"\nQuestion {question_id} at temperature 0.0:")
                print(f"Source: {temp_0_entry.source}")
                print(f"Tactic: {temp_0_entry.tactic}")
                print(f"Time spent: {temp_0_entry.time_spent}")
                # print(f"Submission message: {temp_0_entry.submission_message}")
                print(f"Sequence length: {temp_0_entry.sequence_length}")
                
                # Print first few messages
                # print("\nFirst 3 messages:")
                # for i, msg in enumerate(temp_0_entry.messages[:3]):
                #     if msg.role and msg.body:
                #         print(f"Message {i}: {msg.role} - {msg.body[:50]}...")
