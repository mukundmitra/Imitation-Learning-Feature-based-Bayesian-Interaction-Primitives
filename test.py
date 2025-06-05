## Author: Mukund Mitra (email: mukundmitra@iisc.ac.in)
## This library has been build upon the original work by Joseph Campbell at Arizona State University with contributions from Simon Stepputtis, Geoffrey Clark, Michael Drolet, and Heni Ben Amor. Copyright (c) 2017 Interactive Robotics Lab
## This is the code for testing multiple test trajectories using FBIP, BIP, eBIP and proMP
## Input --> Trained model 'HRI_trained_D3_eBIP.bip' and Testing  'Combined_HRI_D3.csv' last 50 trajectories
## Output --> Predicted trajectory 'predicted_traj_dofs_7_to_18' and MSE and MAE

import pandas as pd
import numpy as np
import intprim
import intprim.util
import intprim.basis
import intprim.filter.spatiotemporal.pf
import sklearn.metrics as sm
 
 
def Test_Imitation_Learning(test_trajectory, trained_model_name, number_of_prediction_steps):
    dof_names = np.array([f"dof {i+1}" for i in range(18)])  # Using only 6 DOFs
    basis_model = intprim.basis.GaussianModel(36, 0.04, dof_names)
    primitive = intprim.BayesianInteractionPrimitive(basis_model)
    primitive.import_data(file_name=trained_model_name)
 
    ## Define scaling groups (to treat each DOF separately)
    scaling_groups_separate = [[i] for i in range(18)]

    selection = intprim.basis.Selection(dof_names, scaling_groups=scaling_groups_separate)
 
    selection.add_demonstration(test_trajectory)
 
    ## Compute observation noise (model MSE)
    # observation_noise = 100* np.diag(selection.get_model_mse(basis_model, np.arange(18)))
    observation_noise = np.diag([0.1] * 6 + [100] * 12)  # Lower noise for DOFs 7-18
 
    prev_observed_index = 0
    active_dofs = np.arange(0, 6)  # Only first 6 DOFs are observed
    predicted_traj_dofs_7_to_18= np.zeros((6, number_of_prediction_steps))  # Predicting DOFs 7-18
 
    for observed_index in range(8, test_trajectory.shape[1], 8):
        inferred_trajectory, _, _, _ = primitive.generate_probable_trajectory_recursive(
            test_trajectory[:, prev_observed_index:observed_index],
            observation_noise,
            active_dofs,
            num_samples=number_of_prediction_steps,
            phase_lookahead=0.15
        )
        prev_observed_index = observed_index
        predicted_traj_dofs_7_to_18= inferred_trajectory[6:, :]  # Extract DOFs 4-6 as predictions
   
    return predicted_traj_dofs_7_to_18
 
if __name__ == "__main__":
    csv_file_path = 'Combined_HRI_D3.csv'
   
    def load_test_trajectories_from_csv(csv_file):
        hdata = pd.read_csv(csv_file)
        hdata.columns = hdata.columns.str.strip()
        traj_counts = hdata['trajectory'].unique()
        test_trajectories = []
        for traj_count in traj_counts:
            traj_data = hdata[hdata['trajectory'] == traj_count]
            dof_columns = [f'dof{i+1}' for i in range(18)]  
            trajectory = np.vstack([traj_data[col].values for col in dof_columns])
            test_trajectories.append(trajectory)
        return test_trajectories
 
    test_trajectories = load_test_trajectories_from_csv(csv_file_path)
 
    # Take the last 100 trajectories for testing
    test_trajectories_sampled = test_trajectories[-50:]
 
    ## Load the pre trained model here
    trained_model_name = 'HRI_trained_D3_eBIP.bip'
    mse_scores = []
    mae_scores = []
 
 
    ## Testing and evaluating
    for i, test_trajectory in enumerate(test_trajectories_sampled):
        original_test_trajectory = test_trajectory[6:, :].copy()  # DOFs 7-18 (ground truth)
        test_trajectory_input = test_trajectory[:6, :]  # DOFs 1-6 (observed)
 
        # Compute the correct number of prediction steps
        number_of_prediction_steps = original_test_trajectory.shape[1]
 
        if number_of_prediction_steps <= 0:
            print(f"Skipping trajectory {i+1} due to zero prediction steps.")
            continue  # Skip trajectories that don’t need prediction
 
        # Zero out DOFs 7-18 before passing into Test_Imitation_Learning
        predicted_zero_trajectory = np.zeros((12, number_of_prediction_steps)) ## 12 = 18 -7 DoFs
 
        # Correctly concatenate the observed and zeroed trajectories
        test_trajectory_modified = np.concatenate((test_trajectory_input, predicted_zero_trajectory), axis=0)
 
        # Call imitation learning function
        predicted_traj_dofs_7_to_18 = Test_Imitation_Learning(test_trajectory_modified, trained_model_name, number_of_prediction_steps)
 
        # Compute MSE and MAE
        mse = sm.root_mean_squared_error(original_test_trajectory, predicted_traj_dofs_7_to_18)
        mae = sm.mean_absolute_error(original_test_trajectory, predicted_traj_dofs_7_to_18)
 
        mse_scores.append(mse)
        mae_scores.append(mae)
 
        print(f"  MSE: {mse}")
        print(f"  MAE: {mae}")
 

    # Print overall statistics
    print(f"\nOverall Performance on {len(test_trajectories_sampled)} test trajectories:")
    print(f"Mean MSE: {np.mean(mse_scores)}, Std MSE: {np.std(mse_scores)}")
    print(f"Mean MAE: {np.mean(mae_scores)}, Std MAE: {np.std(mae_scores)}")