work in progress

```chatinput
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install torch torchvision torchaudio 

# train model
python pythonProject/train.py \
  --data-dir pythonProject/data \
  --class-list pythonProject/class.names \
  --output-dir pythonProject/output \
  --device cpu \
  --learning-rate 6.25e-05 \
  --batch-size 4 \
  --iterations 2000 \
  --checkpoint-period 200 \
  --model "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
  
# plot training vs validation loss
python plot_loss.py

# predict
python predict.py
```