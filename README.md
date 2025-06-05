## Imitation Learning: Feature-based Bayesian Interaction Primitives

This repository contains an imitation learning algorithm Feature-based Bayesian Interaction Primitives (FBIP) for human-robot handover

## Quick Start
Training:

Input: Trajectories from 'Combined_HRI_D3.csv'

Output: Trained model 'HRI_trained_D3_eBIP.bip'
```python
python train.py
```


Testing:

Input: Trained model 'HRI_trained_D3_eBIP.bip' and Testing  'Combined_HRI_D3.csv' last 50 trajectories

Output: Predicted trajectory 'predicted_traj_dofs_7_to_18' and MSE and MAE
```python
python test.py
```

## Dataset Access
If you'd like to access the human-robot handover dataset used for training and evaluation in this project, please email:
mukundmitra@iisc.ac.in (Mukund Mitra)

## Citation
The work has been accepted for publcation at IEEE International Conference on Advanced Robotics and Mechatronics (IEEE ARM 2025), Portmouth, UK.

DOI: To be available soon after publication...


