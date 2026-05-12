import argparse
import json
from src.simulator.evaluation import EvaluationSession
from src.simulator.strategies import STUDY_STRATEGIES


def main():
    parser = argparse.ArgumentParser(description="BrainBot Evaluation")
    parser.add_argument(
        "--strategy",
        choices=list(STUDY_STRATEGIES.keys()),
        required=True,
    )
    parser.add_argument(
        "--task",
        choices=["A", "B"],
        required=True,
    )
    args = parser.parse_args()

    session = EvaluationSession(args.strategy, args.task)
    results = session.run()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
