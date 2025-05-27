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








