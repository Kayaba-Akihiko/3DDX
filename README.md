# Introduction

This is the code repository for the MICCAI 24 paper 3DDX: Bone Surface Reconstruction from a Single Standard-Geometry Radiograph via Dual-Face Depth Estimation.

We provide the loss calculation code at this point due to intellectual property processes.

Abstract:
Radiography is widely used in orthopedics for its affordability and low radiation exposure. 3D reconstruction from a single radiograph, so-called 2D-3D reconstruction, offers the possibility of various clinical applications, but achieving clinically viable accuracy and computational efficiency is still an unsolved challenge.
Unlike other areas in computer vision, X-ray imaging's unique properties, such as ray penetration and standard geometry, have not been fully exploited. 
We propose a novel approach that simultaneously learns multiple depth maps (front and back surfaces of multiple bones) derived from the X-ray image to computed tomography (CT) registration.
The proposed method not only leverages the standard geometry characteristic of X-ray imaging but also enhances the precision of the reconstruction of the whole surface. Our study involved 600 CT and 2651 X-ray images (4 to 5 posed X-ray images per patient), demonstrating our method's superiority over traditional approaches with a surface reconstruction error reduction from 4.78 mm to 1.96 mm and further to 1.76 mm using higher resolution and pretraining.
This significant accuracy improvement and enhanced computational efficiency suggest our approach's potential for clinical application. 
