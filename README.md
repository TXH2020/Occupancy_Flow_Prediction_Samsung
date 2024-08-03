
## Swin Transformer Model:

### Swin Transformer Encoder:

- **Function:** Prepares input image by breaking it into patches and projecting them into a lower-dimensional space.
- **Details:** Processes input data through layers of self-attention and MLPs, capturing spatial dependencies and extracting features.

## Decoder Architecture:

### Pyramid3DDecoder:

- **Function:** Tailored for 3D reconstruction tasks, using convolutional and upsampling layers.
- **Options:** Allows for flexible decoding strategies with parameters like `flow_sep_decode` and `use_pyramid`.

## STrajNet Model:

### STrajNet Class:

- **Function:** Designed for trajectory prediction in autonomous driving.
- **Components:** Integrates Swin Transformer encoder, TrajNet cross-attention, and a custom decoder.
- **Advanced Features:** Supports advanced features like FGMSA for capturing complex patterns.

## SwinTransformer Function:

- **Function:** Convenient interface for instantiating Swin Transformer models.
- **Details:** Allows loading pre-trained weights for transfer learning or fine-tuning.

## Testing Function:

### test_SwinT Function:

- **Function:** Tests the instantiation process of Swin Transformer models.
- **Details:** Configures GPU settings and verifies successful model initialization.



### Layers Overview:

- **PatchEmbed Layer:**

  - **Function:** Breaks input image into patches and projects them into a lower-dimensional space.

- **Conv2D Layer:**

  - **Function:** Performs 2D convolution on input data using learnable filters.

- **LayerNormalization Layer:**

  - **Function:** Applies layer normalization to input data along the feature dimension.

- **TrajNetCrossAttention Layer:**

  - **Function:** Performs cross-attention between trajectory-related inputs and encoded spatial features.

- **Pyramid3DDecoder Layer:**
  - **Function:** Tailored for 3D reconstruction tasks, involving convolutional and upsampling operations.

---

### How to Run the Project:

1. **Clone the Repository:**

   ```
   git clone https://github.ecodesamsung.com/SRIB-PRISM/MSRIT2Elite_Occupancy_Flow_Prediction_for_Automotive_Vision.git
   ```

2. **Navigate to Project Directory:**

   ```
   cd MSRIT2Elite_Occupancy_Flow_Prediction_for_Automotive_Vision
   ```

3. **Install Dependencies:**

   ```
   pip install -r requirements.txt
   ```

4. **Data Preprocessing:**

   ```
   python3 data_preprocessing.py --ids_dir ./Waymo_Dataset/occupancy_flow_challenge/ --save_dir ./Waymo_Dataset/preprocessed_data/ --file_dir ./Waymo_Dataset/tf_example/ --pool 2

  ```

5. **Training:**

   ```
   python3 train.py --save_dir /path/to/save_directory --file_dir /path/to/dataset_directory --model_path /path/to/pretrained_model (optional) --batch_size batch_size --epochs num_epochs --lr learning_rate
   ```

6. **Additional Notes:**
   - Adjust configurations and parameters in the `config.py` file as needed.
   - Ensure that you provide the correct paths for data directories and other options.
   - Refer to the project documentation for more detailed instructions and usage examples.

