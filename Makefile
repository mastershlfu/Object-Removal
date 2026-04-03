ROOT_DIR := $(shell pwd)
export PYTHONPATH := $(PYTHONPATH):$(ROOT_DIR):$(ROOT_DIR)/src

BACKEND_DIR = demo/backend
FRONTEND_DIR = demo/frontend

lama-test:
	cd models/lama && \
	PYTHONPATH=. python bin/predict.py \
	  indir=$(PWD)/data/LaMa_test_images/images \
	  outdir=$(PWD)/outputs/lama_test \
	  model.path=$(PWD)/models/lama/big-lama/
test-mask:
	cd src/pipeline && \
	python test.py

generate-mask:
	cd src/mask && \
	python generate_mask.py

test-trainFasterRCNN:
	cd src/pipeline && \
	python train.py

test-p256-thin-1:
	cd models/RePaint && \
	python test.py --conf_path confs/test_p256_thin.yml --start_idx 0 --end_idx 12000

test-p256-thin-2:
	cd models/RePaint && \
	python test.py --conf_path confs/test_p256_thin.yml --start_idx 12000 --end_idx 24000

test-p256-thin-3:
	cd models/RePaint && \
	python test.py --conf_path confs/test_p256_thin.yml --start_idx 24000 --end_idx 36500

lama-coarse:
	cd src/mask && \
	python generate_sam_mask_and_coarse.py --start 0 --end 12000

lama-coarse2:
	cd src/mask && \
	python generate_sam_mask_and_coarse.py --start 12000 --end 24000

lama-coarse3:
	cd src/mask && \
	python generate_sam_mask_and_coarse.py --start 36302 --end 36441

test-check-repaint:
	cd src/pipeline && \
	python check_repaint.py

inference-fasterRCNN:
	cd src/pipeline && \
	python inference.py

demo-pipeline:
	cd demo && \
	python demo_repaint.py

demoCNN-pipeline:
	cd demo && \
	python demo_CNN.py

train-cnn:
	cd src/refinement && \
	python train.py --epochs 20 --batch_size 16

run-backend:
	@echo "🚀 Starting Backend với PYTHONPATH tại: $(PWD)"
	cd $(BACKEND_DIR) && \
	uvicorn main:app --host 0.0.0.0 --port 8001 --reload

run-frontend:
	@echo "🎨 Starting Frontend tại http://localhost:8000"
	cd $(FRONTEND_DIR) && python -m http.server 8000

