# Internship_PINN_work


First experiment: 
I am finding the core field U from dBr / dt = - nabla (U Br) using  CoreFlowPINN. It has three-layers NN (2x32 → 32x32 → 32x2). I asked for magnetic field data for 2020, 2026 but data was only available till 2020. 
Results: 
The overall loss values decreased year by year (2020 → 2022). This could be due to improvements in the data quality or preprocessing or underlying magnetic field dynamics perhaps got a bit easier to predict using the PINN.
-2022 showed the best initial convergence, but also some instability.
-The network did not collapse or diverge, which is a positive sign of numerical and physical consistency in the PINN formulation.

2) Then I ran with 1000 epoch for 2020, 2021,2022
   The curve
   ![image](https://github.com/user-attachments/assets/5362aaa1-6a7f-44ac-8375-60cd39f624ac)


