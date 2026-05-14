# Models

Model weights are too large for GitHub (50–52MB each).

## Download from Google Drive

| Model | Size | Link |
|---|---|---|
| `weapon_yolov8m_best.pt` | 52MB | [Download](https://drive.google.com/file/d/1_Gen51U5af2S8CCxD-60DcVDlXSKZTb9/view) |
| `fight_best.pth` | 10MB | [Download](https://drive.google.com/file/d/1ogDEq53l-P7ZrqQwBuj7923GgVvgwQjF/view) |
| `yolov8m.pt` | 50MB | [Download](https://drive.google.com/file/d/1qcpfjtwgmL1e-ItGx2MaJDPDwgW9K_Ta/view) |

## After downloading
Place all `.pt` and `.pth` files in this `models/` folder then run:
```bash
python main.py --source data_sample/fight_1.avi --save
```
