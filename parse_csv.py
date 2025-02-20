import csv
import ast
from collections import defaultdict

def compute_average_eval_from_csv(csv_file):
    """
    Parses a CSV file and computes the average eval/squad_exact_match for each origin.

    Parameters:
        csv_file (str): Path to the CSV file.

    Returns:
        dict: A dictionary mapping each origin to its average eval/squad_exact_match.
    """
    origin_totals = defaultdict(lambda: {'sum': 0, 'count': 0})

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        # Skip the header
        next(reader)

        for row in reader:
            if len(row) < 3 or not row[2]:
                continue

            # Parse the dictionary string in the third column
            try:
                data_dict = ast.literal_eval(row[2])
            except (SyntaxError, ValueError):
                print(f"Error parsing row: {row}")
                continue

            for key, metrics in data_dict.items():
                if 'eval/squad_exatct_match' in metrics:
                    origin = key
                    eval_value = metrics['eval/squad_exatct_match']

                    origin_totals[origin]['sum'] += eval_value
                    origin_totals[origin]['count'] += 1

    # Calculate averages
    averages = {
        origin: totals['sum'] / totals['count']
        for origin, totals in origin_totals.items()
        if totals['count'] > 0
    }

    return averages

# Example usage
if __name__ == "__main__":
    csv_file = 'squad_results.csv'  # Update this to your CSV file path

    try:
        averages = compute_average_eval_from_csv(csv_file)

        # Print results
        print("Average eval/squad_exact_match per origin:")
        for origin, avg in averages.items():
            print(f"{origin}: {avg:.2f}")
    except FileNotFoundError:
        print("Error: CSV file not found.")
    except Exception as e:
        print(f"An error occurred: {e}")