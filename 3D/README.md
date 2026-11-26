# 3D

The 2D design applied slice by slice to 3D volumes, with a 3D mask decoder
(`segment_anything/modeling/mask_decoder_3d.py`).

| file | role |
|---|---|
| `train_sam_3d.py` | 3D fine-tuning entry point (paths configured at the top) |
| `infer_3d_3ddec.py` | inference with the 3D decoder (`--cases`, `--output_dir`) |
| `infer_3d_v12.py`, `infer_3d_zs.py` | earlier 2.5D / zero-shot variants, kept for reference |
| `render_3d_check.py` | rendering / sanity check |
| `sacm3d_architecture.png` | architecture figure |

The encoder is the frozen 2D SAM ViT-L; adapters and decoder are the only trained
parts. See the repository root `README.md` for the shared design and results.
