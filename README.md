# light 3dgs unity

Modified version of [UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting) by Aras Pranckevicius.

- Removes SH completely for better performance
- Allows creating multiple assets at once from a folder
- Uses a single triangle as in [this issue](https://github.com/aras-p/UnityGaussianSplatting/pull/181)
- assign camera tags to specific 3DGS objects to avoid unnessecary rendering / sorting calls with multicam setups
