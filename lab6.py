import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def display_results(images, titles, grid):
    plt.figure(figsize=(12, 8))
    for i in range(len(images)):
        plt.subplot(grid[0], grid[1], i + 1)
        if len(images[i].shape) == 2:
            plt.imshow(images[i], cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def bai_1(img_path):
    img = cv2.imread(img_path)
    if img is None: 
        print(f"Khong tim thay: {img_path}")
        return
    img_res = cv2.resize(img, (224, 224))
    img_flip = cv2.flip(img_res, 1)
    M = cv2.getRotationMatrix2D((112, 112), 15, 1.0)
    img_rot = cv2.warpAffine(img_flip, M, (224, 224))
    img_bright = np.clip(img_rot * 1.2, 0, 255).astype(np.uint8)
    img_gray = cv2.cvtColor(img_bright, cv2.COLOR_BGR2GRAY)
    img_norm = img_gray / 255.0
    imgs = [img_res] * 5 + [(img_norm * 255).astype(np.uint8)] * 5
    titles = ["Goc"] * 5 + ["Aug"] * 5
    display_results(imgs, titles, (2, 5))

def bai_2(img_path):
    img = cv2.imread(img_path)
    if img is None: return
    img_res = cv2.resize(img, (224, 224))
    noise = np.random.normal(0, 10, img_res.shape).astype(np.uint8)
    img_noise = cv2.add(img_res, noise)
    M = cv2.getRotationMatrix2D((112, 112), 10, 1.0)
    img_aug = cv2.warpAffine(img_noise, M, (224, 224))
    img_aug = np.clip(img_aug * 1.15, 0, 255).astype(np.uint8)
    img_norm = img_aug / 255.0
    display_results([img_res, (img_norm * 255).astype(np.uint8)], ["Goc", "Aug"], (1, 2))

def bai_3(img_path):
    img = cv2.imread(img_path)
    if img is None: return
    img = cv2.resize(img, (224, 224))
    results = []
    for _ in range(9):
        temp = cv2.flip(img, random.choice([-1, 0, 1]))
        M = cv2.getRotationMatrix2D((112, 112), random.randint(-20, 20), 1.1)
        temp = cv2.warpAffine(temp, M, (224, 224))
        results.append(temp / 255.0)
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(cv2.cvtColor((results[i]*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.show()

def bai_4(img_path):
    img = cv2.imread(img_path)
    if img is None: return
    img_res = cv2.resize(img, (224, 224))
    M = cv2.getRotationMatrix2D((112, 112), -15, 1.0)
    img_rot = cv2.warpAffine(img_res, M, (224, 224))
    img_flip = cv2.flip(img_rot, 1)
    img_bright = np.clip(img_flip * 0.8, 0, 255).astype(np.uint8)
    img_gray = cv2.cvtColor(img_bright, cv2.COLOR_BGR2GRAY)
    img_norm = img_gray / 255.0
    imgs = [img_res] + [(img_norm * 255).astype(np.uint8)] * 3
    titles = ["Goc", "Aug 1", "Aug 2", "Aug 3"]
    display_results(imgs, titles, (1, 4))

if __name__ == "__main__":
    p1 = "can_ho.jpeg"
    p2 = "oto.jpeg"
    p3 = "phong.jpeg"
    p4 = "trai_cay.jpeg"
    
    bai_1(p1)
    bai_2(p2)
    bai_3(p3)
    bai_4(p4)