# Internship_PINN_work


First experiment: 
I am finding the core field U from dBr / dt = - nabla (U Br) using  CoreFlowPINN. It has three-layers NN (2x32 → 32x32 → 32x2). I asked for magnetic field data for 2020, 2026 but data was only available till 2020. 
1) At first I ran with 10 epochs for (2020 → 2022) 
Results: 
The overall loss values decreased year by year. This could be due to improvements in the data quality or preprocessing or underlying magnetic field dynamics perhaps got a bit easier to predict using the PINN.
-2022 showed the best initial convergence, but also some instability.
-The network did not collapse or diverge, which is a positive sign of numerical and physical consistency in the PINN formulation.

2) Then I ran with 1000 epoch for 2020, 2021,2022
   The loss curve
   ![image](https://github.com/user-attachments/assets/5362aaa1-6a7f-44ac-8375-60cd39f624ac)

Observation: The model for 2022 showed the best learning with a quick drop in loss and stable training, meaning it was able to fit the data well. 2020 had a sudden spike at the end, which means the training became unstable. 2021 stayed mostly flat with little improvement, which suggests the model didn’t learn much.

3) Then I ran with 10000 epoch for 2020, 2021,2022
   The loss curve
![image](https://github.com/user-attachments/assets/68000f3c-411e-44b4-ba23-9cd139f7d854)


| Year     | Previous Graph (1k Epochs)                     | New Graph (10k Epochs)                                      |
| -------- | ---------------------------------------------- | ----------------------------------------------------------- |
| **2020** | Sudden spike near the end → instability        | Smoother curve, no spike, more stable   |
| **2021** | Almost flat throughout → no learning           | Still flat, but now clearer that it's slowly learning       |
| **2022** | Fast drop, small increase → likely overfitting | Still fast drop, but then stabilizes with a bit fluctuations |


After using L1 and L2 together

epoch: 1000

![image](https://github.com/user-attachments/assets/9ad9e2d8-291d-402e-b2ca-f3c90cca162d)



Issues :

1) We came to know that autograd is not taking Br into account while differentiating and treating it like constant
2) I filtered thetas with a range (50°, 70°) in degrees, but the grid was in radians.
3) I faced problems regarding shape: The size of tensor a (7200) must match the size of tensor b (7) at non-singleton dimension 0 means that u_theta (or u_phi) has shape [7200, 1] but Br has shape [7, 1]. That suggests my neural network is correctly producing one prediction per [theta, phi] pair (7200 total), but the input data Br is not matching in shape. This happens because I am flattened the mf_grid, but it likely has wrong dimensions (like shape (1, 7) or (7,)), meaning only 7 values instead of 7200 (which would be 20 × 360 if your theta and phi filters work correctly).
To fix the thirds issue:
Remembered pytorch takes 2D shape: [batch_size, num_features]
So I just sliced everything to make sure all are same 
example:
mf_grid shape: (20, 360) 
sv_grid shape: (20, 360)

20 values of theta (colatitude) and 360 values of phi (longitude)

the number of grid points 20 × 360 = 7200 
now I have shape of all [7200, 1] which is [batch_size, num_features]

In other words, I have 7200 input points, and for each point, 1 value (like theta or Br)vThis is exactly what PyTorch models expect — and why we always flatten and reshape that way.


Updated code:

Update:


with  lemda 10 and 
Parameters:
learning rate : 0.001
node_inputs = 2
node_outputs = 2
node_layer = 64
hidden_layers = 3

Result
 Epoch 987 --- Loss 1487.868286--- Loss_L1 1487.867554--- Loss_L2 0.000077
 Epoch 988 --- Loss 1485.363281--- Loss_L1 1485.362549--- Loss_L2 0.000077
 Epoch 989 --- Loss 1483.214844--- Loss_L1 1483.214111--- Loss_L2 0.000077
 Epoch 990 --- Loss 1481.253662--- Loss_L1 1481.252930--- Loss_L2 0.000077
 Epoch 991 --- Loss 1479.261719--- Loss_L1 1479.260986--- Loss_L2 0.000077
 Epoch 992 --- Loss 1477.135254--- Loss_L1 1477.134521--- Loss_L2 0.000077
 Epoch 993 --- Loss 1474.908813--- Loss_L1 1474.908081--- Loss_L2 0.000077
 Epoch 994 --- Loss 1472.700684--- Loss_L1 1472.699951--- Loss_L2 0.000077
 Epoch 995 --- Loss 1470.617798--- Loss_L1 1470.617065--- Loss_L2 0.000077
 Epoch 996 --- Loss 1468.672607--- Loss_L1 1468.671875--- Loss_L2 0.000077
 Epoch 997 --- Loss 1466.808350--- Loss_L1 1466.807617--- Loss_L2 0.000077
 Epoch 998 --- Loss 1464.942261--- Loss_L1 1464.941528--- Loss_L2 0.000077
 Epoch 999 --- Loss 1463.019043--- Loss_L1 1463.018311--- Loss_L2 0.000077


 After changing  params, this is the best combination 

![image](https://github.com/user-attachments/assets/298b194a-2ae4-4eab-9e44-9bc602db6463)


![image](https://github.com/user-attachments/assets/60da10c2-29f5-45bd-b710-e2a7ba72a617)

After
lambda_values = [0, 0.1, 1, 10, 100]


| λ   | Total Loss ↓ | L1 (Data Fit) ↓ | L2 (Constraint) | Behavior Summary                          |
| --- | ------------ | --------------- | --------------- | ----------------------------------------- |
| 0   | **964.77**   | 964.77          | \~0.0001        | ignores l2 
| 0.1 | 1731.73      | 1731.73         | \~0.0001        |  
| 1   | 2641.22      | 2641.22         | \~0.0001        | 
| 10  | 1955.09      | 1955.09         | \~0.0001        | The total loss still converges well (~20 million → ~1955).       
| 100 | 2179.31      | 2179.30         | \~0.0001        | Starts to slow convergence slightly. Between epoch 600–700, loss increases

10 is best trade off

I tried different learning rates but i can see spiles in 
![image](https://github.com/user-attachments/assets/25afcdde-06fa-4031-b5b7-60af5fba05d6)



I think we have good enough combination now.
learning rate is best 5e-4 instead of 0.001 since there are no spikes
![image](https://github.com/user-attachments/assets/25605b4a-46b7-4eea-84fe-136ba90e31e6)



They saw that L2 was too small by default, and they scaled it not arbitrarily, but to balance the gradient sizes. 


3rd june

 I struggled at first with how the patch-specific data like theta, phi grids, and magnetic field values were handled. Initially, these inputs were outside the training loop, which caused confusion and errors because the model wasn’t receiving the correct data for each patch during training.

After some debugging and clarification, I moved all the patch-specific tensor creation (thetas_nn, phis_nn, Br_nn, dBrdt_nn, dBrdth_nn, dBrdph_nn) inside the training loop. This way, for every patch and every training iteration, the model gets the exact inputs it needs, making the training consistent and correct. and passed the params to compute function where all my calculations are. This helped me ensure that the radial induction equation and quasi-geostrophic constraints are properly enforced in the loss.

Overall today i did these:
1) At the CMB, the SV varies between ±20 µT/year. By keeping r in kilometers, rather than converting it into meters, the total loss could be squared to µT/year. How does this help?
I kept r in kilometers

2) How to check if the loss on the grid is homogeneous or heterogeneous?
I plotted the loss on the spatial grid to visualize where errors are higher or lower, which helps understand model performance spatially.

3) How to generate and average multiple realizations for the same square (with different initial conditions)?
I trained the model multiple times on the same patch with different random initializations and averaged the loss results to get a good estimate.

4) How to compute for the whole map by solving for several overlapping small squares and wrap everything at the end?
I tiled the globe (avoiding poles and equator) with overlapping small patches, trained separately on each, then planned to stitch the patch results to form a global solution.















