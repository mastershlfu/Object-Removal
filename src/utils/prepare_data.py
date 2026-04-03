import os 
from ultralytics import YOLO
from tqdm import tqdm

# prepare data for training edge generator, avoid img with high ratio of human, cats, dogs

IMG_DIR = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/data/Places/val_large"
TXT_PATH = "/home/ml4u/BKTeam/source/BaoNhi/Object-Removal/data/Places/clean_places2.txt"

def filter_dataset(img_dir, output_txt, max_person_ratio):
    model = YOLO("yolov8x-seg.pt")

    all_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"found {len(all_files)} imgs.")

    valid_count = 0

    with open(output_txt, 'w') as f:
        for filename in tqdm(all_files, desc="reading igm"):
            img_path = os.path.join(img_dir, filename)

            # avoid human, cats, dogs
            results = model.predict(img_path, classes=[0, 15, 16], verbose=False)
            result = results[0]

            img_h, img_w = result.orig_shape
            img_area = img_w*img_h
            
            is_valid = True

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                person_area = (x2 - x1)*(y2 - y1)
                ratio = person_area/img_area

                if ratio > max_person_ratio:
                    is_valid = False
                    break
            
            if is_valid:
                abs_path = os.path.abspath(img_path)
                f.write(abs_path +  '\n')
                valid_count += 1
    
    print("done")
    print(f"Keep: {valid_count} valid image.")
    print(f"Drop: {len(all_files) - valid_count} image with high ratio target's area.")
    print(f"File saved at: {output_txt}")

if __name__ == "__main__":
    img_dir = IMG_DIR
    
    # output path
    txt_path = TXT_PATH
    # with open(txt_path, "r") as f1, open(TXT_PATH, "w") as f2:
    #     f2.write(f1.read())

    filter_dataset(img_dir, txt_path, max_person_ratio=0.25)