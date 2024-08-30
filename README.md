# table-kv-extractor
Key Value Extraction from Tables in Documents

Divided into 3-stage approach:

### 1. Table Detection
   Input: document page image
   Output: Detected tables from where K, and V need to be extracted

### 2. Table Structure Recognition
   Input: Detected Table region
   Output: Bounding boxes of rows, columns, and cells with their row and columns mapped

### 3. Table Content Recognition
   Input: Detected Cell bounding boxes
   Output: OCR of the cells to be used as keys and values

More information available [here](https://docs.google.com/document/d/195hJvzB7FHCWJIyQwvkvLEL6n6vO85EvYp70we2vwGc/edit)
