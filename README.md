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

3) Then I ran with 1000 epoch for 2020, 2021,2022
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



