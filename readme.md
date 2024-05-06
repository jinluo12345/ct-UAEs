# ct-UAEs

A brief description of the ct-UAEs project, focusing on its objectives and the value it provides.

## Step 1: Download the Dataset

The datasets required for the ct-UAEs project are available via the following Google Drive links. Please download them as part of the initial setup:

- **mp_13 Dataset**: [Download here](https://drive.google.com/file/d/1u1n_CoPfVJVbtXr9Y8mONkv3hy6_fHTM/view?usp=sharing)
- **mp Dataset**: [Download here](https://drive.google.com/file/d/1RxDl48_MfMWpIMvgGcGX2pI9ZMusR6Dl/view?usp=drive_link)

Refer to the `download.txt` for additional guidance on downloading these datasets.

## Step 2: Extract and Organize Datasets

Once downloaded, extract each dataset to the project directory under their respective names (`mp` and `mp_13`). Ensure they are placed according to the following structure:
```csharp
ct-UAEs/
│
├── ...
├── ckpt/ # Corresponding ct-UAEs 
├── ct/ # Main source code directory
├── mp/ # Extracted mp dataset
├── mp_13/ # Extracted mp_13 dataset
├── ...
└── download.txt
```
This organization is crucial for ensuring that the datasets are correctly recognized by the project scripts.

## Next Steps

- **Set up your development environment**: Ensure all necessary libraries and dependencies are installed. You can run 
```
pip install -r requirements.txt
```
- **Begin model training**:execute the provided scripts to start training your models.

## How to Run

To train the models for single or multi-task learning, follow the steps below:

### Single-Task Training

For single-task training, execute the following command:

```bash
python train\\main.py --load mp_13 --file mp_13_1_f.csv
```
Replace mp_13 with mp and file with mp_formation.csv if you prefer to use the mp dataset. The --file flag specifies the CSV file containing the dataset. Here, f represents the formation energy (f) property.

### Multi-Task Training
For multi-task training with different combinations of properties, execute the corresponding script. For example, for formation energy (f) and bandgap (b), use:
```bash
python train\\main_mt_2.py --load mp_13 --file mp_13_2_fb.csv
```
For formation energy (f), bandgap (b), and total energy (e), use:
```bash
python train\\main_mt_3.py --load mp_13 --file mp_13_3_fbe.csv
```
For all four properties (formation energy (f), bandgap (b), total energy (e), and total magnetization (m)), use:
```bash 
python train\\main_mt_4.py --load mp_13 --file mp_13_4_fbem.csv
```
## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

<!-- ## License

This project is licensed under the MIT License. -->

## Acknowledgments

We acknowledge the contributors and datasets used in this project for their valuable contributions to scientific research.
