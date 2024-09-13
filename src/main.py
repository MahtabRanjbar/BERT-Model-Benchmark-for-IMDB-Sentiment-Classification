import yaml
from utils import setup_logger
from data_processing import load_imdb_data, prepare_dataset
from benchmark import run_benchmark, plot_benchmark_results, save_benchmark_results
from trainer import create_trainer, compute_metrics
from models import MODELS


def load_config(config_path):
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)


def main():
    logger = setup_logger("imdb_classification", "imdb_classification.log")
    logger.info("Starting IMDB classification benchmark")

    # Load configuration
    config = load_config('config/config.yaml')

    # Load and prepare data
    imdb_data = load_imdb_data()
    train_dataset, eval_dataset = prepare_dataset(imdb_data)

    # Run benchmark
    results_df = run_benchmark(MODELS, train_dataset, eval_dataset, create_trainer, compute_metrics,config)

    # Analyze and save results
    plot_benchmark_results(results_df)
    save_benchmark_results(results_df)

    logger.info("Benchmark completed. Results saved to benchmark_results.csv and benchmark_results.png")

if __name__ == "__main__":
    main()