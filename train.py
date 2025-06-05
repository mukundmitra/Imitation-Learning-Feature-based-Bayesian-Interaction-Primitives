## Author: Mukund Mitra (email: mukundmitra@iisc.ac.in)
## This library has been build upon the original work by Joseph Campbell at Arizona State University with contributions from Simon Stepputtis, Geoffrey Clark, Michael Drolet, and Heni Ben Amor. Copyright (c) 2017 Interactive Robotics Lab
## This code is used for training FBIP, BIP, eBIP and ProMP models
## Input --> Trajectories from 'Combined_HRI_D3.csv'
## Output --> Trained model 'HRI_trained_D3_eBIP.bip'

import pandas as pd
import intprim.basis
import intprim.filter
import numpy as np
import copy
import intprim
import sklearn.mixture

def Training_Imitation_Learning(training_trajectories):
    ##scaled together
    # scaling_groups_together = [[i for i in range(18)]]  # All 18 DoFs scaled together
    scaling_groups_separate = [[i] for i in range(18)]  # Each DoF scaled separately


    # Get phase velocity mean and variance
    phase_velocity_mean, phase_velocity_var = intprim.examples.get_phase_stats(training_trajectories)
    phase_accel_mean = 1e-4
    phase_accel_var = 1e-10
    dof_names = np.array([f"dof {i+1}" for i in range(18)])

    ## Create MixtureModel using Gaussian and Sigmoidal basis models.
    ## Example below shows first 6 DoFs with Gaussian and next 6 DoFs with polynomial
    # basis_model_gaussian = intprim.basis.GaussianModel(8, 0.1, dof_names[:6])  # For first 6 DoF
    # basis_model_polynomial = intprim.basis.PolynomialModel(8, dof_names[6:])  # For last 6 DoF
    # models = [basis_model_gaussian, basis_model_polynomial]
    # basis_model_mixture = intprim.basis.MixtureModel(models)

    #Create basis gaussian model
    basis_model_gaussian = intprim.basis.GaussianModel(36, 0.04, dof_names) ## 36 is the number of basis functions
    

    selection = intprim.basis.Selection(dof_names, scaling_groups =scaling_groups_separate)
    for trajectory in training_trajectories:
        selection.add_demonstration(trajectory) 

    # Create BayesianInteractionPrimitive using the Basis Model
    primitive = intprim.BayesianInteractionPrimitive(
        basis_model=basis_model_gaussian,
        scaling_groups=scaling_groups_separate  # Apply the new scaling groups
    )

    # Compute standardization for scaling
    for trajectory in training_trajectories:
        primitive.compute_standardization(trajectory)

    # Add demonstrations to the primitive
    for trajectory in training_trajectories:
        primitive.add_demonstration(trajectory)
    

    # Get mean and covariance of basis weights
    mean, cov = primitive.get_basis_weight_parameters()

    ## The above (mean, cov) is used for extended kalman filter 
    ##---------------For extended kalman filter (uncomment the below lines of code)-----------------------
    # ensemble_sampler = sklearn.mixture.GaussianMixture(n_components = 2, reg_covar = 1e-4)
    # ensemble_sampler.fit(primitive.basis_weights)
    # num_train_trajectories = 400
    # sampled_ensemble = ensemble_sampler.sample(num_train_trajectories)[0]

    # # Initialize Extended Kalman Filter (BIP)
    # filter_extendedKF = intprim.filter.spatiotemporal.ExtendedKalmanFilter(
    #     basis_model=basis_model_gaussian,
    #     initial_phase_mean=[0.0, phase_velocity_mean],
    #     initial_phase_var=[1e-4, phase_velocity_var],
    #     proc_var=1e-6,
    #     mean_basis_weights=mean,
    #     cov_basis_weights=cov,
    # )

 
    ##---------------For ensemble kalman filter-----------------------
    # Initialize Ensemble Kalman Filter(eBIP)
    filter_ensembleKF = intprim.filter.spatiotemporal.EnsembleKalmanFilter(
        basis_model=basis_model_gaussian,
        initial_phase_mean=[0.0, phase_velocity_mean],
        initial_phase_var=[1e-4, phase_velocity_var],
        proc_var=1e-6,
        # initial_ensemble=sampled_ensemble,
        initial_ensemble=primitive.basis_weights

    )

    ##---------------For kalman filter (uncomment the below lines of code)-----------------------
    # # Initialize Kalman Filter(Probabilistic movement primitives,(proMP))(Gaussian model)
    # filter_kf = intprim.filter.KalmanFilter(
    #     basis_model=basis_model_gaussian,
    #     mean_basis_weights=mean,
    #     cov_basis_weights=cov,
    #     align_func =intprim.filter.align.dtw.fastdtw,
    #     iterative_alignment=False
    # )


    ##---------The below code uses ensemble kalman filter (eBIP)----------------
    ## Change the filter accordingly
    primitive.set_filter(copy.deepcopy(filter_ensembleKF))
    return primitive



if __name__ == "__main__":
    def load_train_trajectories_from_csv(csv_file):
        hdata = pd.read_csv(csv_file)
        hdata.columns = hdata.columns.str.strip()  
        traj_counts = hdata['trajectory'].unique()  
        train_trajectories = []
        for traj_count in traj_counts:
            traj_data = hdata[hdata['trajectory'] == traj_count]
            dof_columns = [f'dof{i+1}' for i in range(18)]  # Select all 12 DoF columns
            trajectory = np.vstack([traj_data[col].values for col in dof_columns])
            train_trajectories.append(trajectory)
        return train_trajectories

    # Import training data
    csv_file_path = 'Combined_HRI_D3.csv'
    train_trajectories = load_train_trajectories_from_csv(csv_file_path)

    # Remove last 2 trajectories for testing
    train_trajectories = train_trajectories[:-50]

    # Train the Imitation Learning model
    primitive = Training_Imitation_Learning(train_trajectories)

    # Export the trained model to a .bip file
    primitive.export_data('HRI_trained_D3_eBIP.bip')
