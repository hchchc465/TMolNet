#
# def objective(trial):
#     # Define hyperparameter sampling based on provided ranges
#     hyperparams = {
#         "gnn_num_layers": trial.suggest_categorical("gnn_num_layers", [2,4,6]),
#         "seq_num_layers": trial.suggest_categorical("seq_num_layers", [2,4,6]),
#         #"Geo_num_layers": trial.suggest_int("Geo_num_layers", [2,4,6]),
#         "lr": trial.suggest_categorical("lr", [1e-3,3e-3,6e-3,9e-3, 1e-2, 3e-2, 6e-2, 9e-2]),
#         "dropout": trial.suggest_float("dropout", 0.2, 0.5),
#         "alpha": trial.suggest_float("alpha", 0.01, 0.2),
#         "beta": trial.suggest_float("beta", 0.01, 0.2),
#         "batch_size": trial.suggest_categorical("batch_size", [8,16,32,64])
#     }
#
#     # Get arguments and update with sampled hyperparameters
#     arg = get_args()
#     for key, value in hyperparams.items():
#         setattr(arg, key, value)
#
#     # Run the main function and return the validation metric
#     val_metric = main(arg)
#     return val_metric
#
#
# if __name__ == '__main__':
#
#     # Create Optuna study to maximize or minimize the objective (adjust direction as needed)
#     study = optuna.create_study(direction="minimize")  # Use "minimize" if optimizing for loss
#     study.optimize(objective, n_trials=50)  # Run 50 trials, adjust as needed
#
#     # Print best trial results
#     print("Best trial:")
#     trial = study.best_trial
#     print(f"  Value: {trial.value}")
#     print("  Params: ")
#     for key, value in trial.params.items():
#         print(f"    {key}: {value}")
#
#     Optionally, run main with the best hyperparameters
#     arg = get_args()
#     for key, value in trial.params.items():
#         setattr(arg, key, value)
#     main(arg)