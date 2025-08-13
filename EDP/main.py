import argparse, os, pickle
import pandas as pd
from utils import evaluate_cpts, calculate_results, model_name_translation
from learner import BayesianNetworkLearner

def main():
    parser = argparse.ArgumentParser(description='Learn Bayesian Network probabilities from BIF models')
    parser.add_argument('--save_BNs', default=False,type=bool, help='Save the Bayesian Networks to file')
    parser.add_argument('--output', default='./', help='Output directory for results')
    parser.add_argument('--learning_method',nargs='+', default=['Counting','Dirichlet'],choices=['Dirichlet','Counting'], help='Learning method to use')
    parser.add_argument('--prior_weight_hyper_parameter', default=2, type=float, help='Alpha weight for prior probabilities')
    parser.add_argument('--sampling_method', default='per_model_by_parameter', choices=['per_node_and_parents','per_node_and_parents_by_size', 'per_model_by_parameter'], help='Sampling method to use')
    parser.add_argument('--models', dest='models', nargs='+', default=['GPT-4o (SepState)',"Uniform"],help="Specify one or more models.")

    args = parser.parse_args()
    args.models = [model_name_translation[name] for name in args.models]

    all_results = []
    os.makedirs(args.output, exist_ok=True)
    sample_num_list = [3,10,30,100,300,1000,10000]
    if not args.sampling_method == 'per_model_by_parameter': sample_num_list = list(range(2,16))
    for samples in sample_num_list:
        print(f"\nProcessing samples: {samples}, prior_weight_hyper_parameter: {args.prior_weight_hyper_parameter}, with sampling method: {args.sampling_method}")
        for LLM_model in args.models:
            for learning_method in args.learning_method:
                learner = BayesianNetworkLearner(sample_num=samples, prior_weight_hyper_parameter=args.prior_weight_hyper_parameter, llm_model_name=LLM_model, sample_method=args.sampling_method, output_dir=args.output, save_BNs=args.save_BNs, learning_method=learning_method,args=args)
                for model in learner.load_models():
                    print(f"\nProcessing model: {model.name}")
                    pickle_file = f"NodeSamplingCache/{samples}_{model}_{args.sampling_method}_shared_samples.pkl"
                    if os.path.exists(pickle_file):
                        with open(pickle_file, 'rb') as f:
                            sample_data = pickle.load(f)
                    else:
                        sample_data = learner.generate_samples(model, samples, args.sampling_method)
                        with open(pickle_file, 'wb') as f:
                            pickle.dump(sample_data, f)

                    initial_cpts_used, learned_from_data, combined_cpts = learner.learn_cpt_probabilities(sample_data,model)

                    metrics_initial, _ = evaluate_cpts(model, initial_cpts_used)
                    metrics_learned, _ = evaluate_cpts(model, learned_from_data)
                    metrics_combined, _ = evaluate_cpts(model, combined_cpts)

                    all_results.append(calculate_results(model, samples, args.prior_weight_hyper_parameter, args.sampling_method, LLM_model, metrics_initial, metrics_learned, metrics_combined, learning_method))

    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(args.output, f'comparison_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

if __name__ == "__main__":
    main()