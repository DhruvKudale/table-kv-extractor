import os

import cv2
from PIL import Image
import pandas as pd
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import argparse
from td import TableDetector
from tsr import get_rows_from_yolo, get_cols_from_tatr, get_cells_from_rows_cols
from utils import *
from bs4 import BeautifulSoup
import json

# Set the TESSDATA_PREFIX environment variable
# os.environ['TESSDATA_PREFIX'] = '/raid/ganesh/vishak/miniconda3/envs/coe/share'


# parser = argparse.ArgumentParser(description='Table Detection')
# parser.add_argument('-p', '--pdf', type=str, help='Path to the image', required=True)
#
# args = parser.parse_args()

def save_cells(cropped_img, cells, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for row_idx, row in cells.items():
        for col_idx, cell in enumerate(row):
            x1, y1, x2, y2 = cell
            cell_img = cropped_img[y1:y2, x1:x2]
            output_path = os.path.join(output_dir, f'cell_{row_idx}_{col_idx + 1}.png')
            plt.imsave(output_path, cell_img)
            # print(f'Saved {output_path}')


def ocr_cells(cropped_img, cells):
    ocr_data = []
    for row_idx, row in cells.items():
        for col_idx, cell in enumerate(row):
            x1, y1, x2, y2 = cell
            cell_img = cropped_img[y1:y2, x1:x2]
            # Convert cell image to PIL Image for OCR
            cell_pil_img = Image.fromarray(cell_img)
            ocr_result = pytesseract.image_to_string(cell_pil_img, config='--psm 6')
            ocr_result.replace("\n", " ")
            # print(ocr_result)
            ocr_data.append((row_idx, col_idx, ocr_result.strip()))
    return ocr_data


def get_updated_html_output(cells, html_string):
    soup = BeautifulSoup(html_string, 'html.parser')
    trs = soup.find('tbody').find_all('tr')
    for r, tr in enumerate(trs):
        tds = tr.find_all('td')
        for c, td in enumerate(tds):
            bbox = cells[r + 1][c]
            bbox = map(str, bbox)
            bbox_attr = "bbox " + ' '.join(bbox)
            td['title'] = bbox_attr
    return soup



if __name__=="__main__":
    
    table_det = TableDetector()
    # images = pdf_to_images(args.pdf)
    # image = np.array(images[0])
    image_file = 'td/samples/sample3.jpeg'
    image = cv2.imread(image_file)
    # plt.imsave("image.jpg", image)
    dets = table_det.predict(image=image)
    all_ocr_data = []
    for det in dets:
        x1, y1, x2, y2 = map(int, det)  # Convert coordinates to integers
        cropped_img = image[y1:y2, x1:x2]  # Crop the image using the bounding box
        plt.imsave("cropped_img.jpg", cropped_img)
        img_file = "cropped_img.jpg"
        #rows, cols = get_rows_cols_from_tatr(img_file)
        rows = get_rows_from_yolo(img_file)
        cols = get_cols_from_tatr(img_file)
        # print(len(rows))
        # print(len(cols))

        rows, cols = order_rows_cols(rows, cols)

        ## Visualize Rows and Columns
        row_image = draw_bboxes(img_file, rows, color = (255, 66, 55), thickness = 2)
        cols_image = draw_bboxes(img_file, cols, color= (22, 44, 255), thickness = 2)
        cv2.imwrite('rows.jpg', row_image)
        cv2.imwrite('cols.jpg', cols_image)


        ## Extracting Cells

        cells = get_cells_from_rows_cols(rows, cols)
        print(cells)

        ## Visualize Extracted Cells
        all_cells = []
        for kr in cells.keys():
            all_cells += cells[kr]
        # cell_image = draw_bboxes(img_file, all_cells, color = (23, 255, 45), thickness = 1)
        # cv2.imwrite('cell.jpg', cell_image)
        # print("Cells: ", cells)
        save_cells(cropped_img, cells, "./cells")

        ## OCR on Cells
        ocr_data = ocr_cells(cropped_img, cells)
        all_ocr_data.extend(ocr_data)

    # Convert OCR data to a pandas DataFrame
    df = pd.DataFrame(all_ocr_data, columns=['Row', 'Column', 'Text'])
    df = df.pivot(index='Row', columns='Column', values='Text')
    #print(df)
    # Save DataFrame to a CSV file
    #df.to_csv('./extracted_table.csv', index=True)
    df_list = df.values.tolist()

    # Make key value pairs
    encoded_json = []

    for r in range(len(rows)):
        for c in range(0, len(cols), 2):
            item = {}
            item['row'] = r + 1
            item['col_of_key'] = c + 1
            item['col_of_val'] = c + 2
            item['key'] = df_list[r][c]
            try:
                item['value'] = df_list[r][c + 1]
            except:
                item['value'] = ''
            item['key_bbox'] = cells[r + 1][c]
            try:
                item['val_bbox'] = cells[r + 1][c + 1]
            except:
                item['val_bbox'] = ''
            encoded_json.append(item)

    # Writing to sample.json
    with open("sample.json", "w") as outfile:
        json.dump(encoded_json, outfile)

    html_string = df.to_html()
    # print(html_string)

    hocr_string = get_updated_html_output(cells, html_string)
    print(hocr_string)
    print(encoded_json)
    # You need hocr_string and encoded_json


